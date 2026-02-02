

import torch
import torch.nn as nn
import torch.nn.init as init
import lightning
from torch.nn import functional as F
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from timm.scheduler.step_lr import StepLRScheduler


from EvEye.model.DavisEyeEllipse.EPNet.Backbone.MobileNetV3Backbone import MobileNetV3Backbone
from EvEye.model.DavisEyeEllipse.EPNet.Head.EPHead_v2 import EPHead_v2
from EvEye.model.DavisEyeEllipse.EPNet.Loss import CtdetLoss
from EvEye.model.DavisEyeEllipse.EPNet.Metric import p_acc, cal_mean_distance
from EvEye.model.DavisEyeEllipse.EPNet.Predict import soft_argmax_decode

from EvEye.model.DavisEyeEllipse.EPNet.CEMR import CEMR

LOSS_WEIGHT = {
    "hm_weight": 1.0,
    "coord_weight": 1.0,
    "ab_weight": 1.0,
    "trig_weight": 1.0,
    "iou_weight": 15.0,
}

HEAD_DICT_INTEGRAL = {
    "hm_L": 1, "ab_L": 2, "trig_L": 2, "mask_L": 1,
    "hm_R": 1, "ab_R": 2, "trig_R": 2, "mask_R": 1,
}


class EPNet(lightning.LightningModule):
    def __init__(self,
                 input_channels=4,
                 head_dict=HEAD_DICT_INTEGRAL,
                 loss_weight=LOSS_WEIGHT,
                 fpn_out_channels=64,
                 temp_start=1.0,
                 temp_end=0.1,
                 temp_epochs=50,
                 wing_w=5.0,
                 wing_epsilon=1.0,
                 **kwargs):
        super(EPNet, self).__init__()
        self.save_hyperparameters()

        self.criterion = CtdetLoss(loss_weight=loss_weight, wing_w=self.hparams.wing_w,
                                   wing_epsilon=self.hparams.wing_epsilon)

        self.backbone = MobileNetV3Backbone(input_channels=input_channels)

        self.in_filters = self.backbone.in_filters

        self.cemr_decoder = CEMR(
            in_channels_list=self.in_filters,
            out_channels=fpn_out_channels
        )

        self.head = EPHead_v2(in_channels=fpn_out_channels, head_conv=512, head_dict=head_dict)

        self._initialize_weights()
        self.current_temp = self.hparams.temp_start

        self.validation_outputs = []

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight);
                init.constant_(m.bias, 0)

    def forward(self, x):
        p3_feat, p4_feat, p5_feat = self.backbone(x)
        p3_out, p4_out, p5_out = self.cemr_decoder(p3_feat, p4_feat, p5_feat)
        return self.head(p3_out)

    def on_train_epoch_start(self):

        progress = min(1.0, self.current_epoch / self.hparams.temp_epochs)
        self.current_temp = self.hparams.temp_start - (self.hparams.temp_start - self.hparams.temp_end) * progress
        self.criterion.set_temperature(self.current_temp)
        self.log("temperature", self.current_temp, on_step=False, on_epoch=True, prog_bar=True)

    def _log(self, name, metric):
        self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        pred = self(batch["input"])
        total_loss_L, total_loss_R, loss_show = self.criterion(pred, batch)

        if total_loss_L is None or total_loss_R is None: return None

        # Dynamic Weighting
        alpha = 3.0
        loss_sum_detached = total_loss_L.detach() + total_loss_R.detach()

        if loss_sum_detached > 1e-8:
            normalized_loss_L, normalized_loss_R = total_loss_L.detach() / loss_sum_detached, total_loss_R.detach() / loss_sum_detached
        else:
            normalized_loss_L, normalized_loss_R = torch.tensor(0.5, device=self.device), torch.tensor(0.5,
                                                                                                       device=self.device)

        weight_L, weight_R = torch.exp(alpha * normalized_loss_L), torch.exp(alpha * normalized_loss_R)
        loss = (weight_L * total_loss_L + weight_R * total_loss_R) / (weight_L.detach() + weight_R.detach())

        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for key, value in loss_show.items(): self._log(f"train_{key}", value)
        self._log("train_loss_L_unweighted", total_loss_L);
        self._log("train_loss_R_unweighted", total_loss_R)
        self._log("train_weight_L", weight_L);
        self._log("train_weight_R", weight_R)

        total_batches = self.trainer.num_training_batches
        if (batch_idx == 0): print(
            f"\n--- Epoch {self.current_epoch}: Current Temperature = {self.current_temp:.4f} ---")
        if (batch_idx == 0 or batch_idx == total_batches // 2 or batch_idx == total_batches - 1):
            print(f"\n--- Epoch {self.current_epoch}, Batch {batch_idx + 1}/{total_batches} ---")
            print(
                f"Step {self.global_step}: Final Loss = {loss.item():.4f} | L_unw: {total_loss_L.item():.4f} | R_unw: {total_loss_R.item():.4f} | W_L: {weight_L.item():.4f} | W_R: {weight_R.item():.4f}")
        return loss


    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx):
        pred = self(batch["input"])
        total_loss_L, total_loss_R, _ = self.criterion(pred, batch)

        if total_loss_L is None or total_loss_R is None: return None
        loss = (total_loss_L + total_loss_R) / 2.0

        final_temp = self.hparams.temp_end
        center_left_pred, center_right_pred = soft_argmax_decode(pred['hm_L'],
                                                                 temperature=final_temp), soft_argmax_decode(
            pred['hm_R'], temperature=final_temp)

        center_left_gt, close_left = batch["center_L"], batch["close_L"]
        center_right_gt, close_right = batch["center_R"], batch["close_R"]

        dets_left = {'xs': center_left_pred.squeeze(1)[:, 0:1], 'ys': center_left_pred.squeeze(1)[:, 1:2]}
        dets_right = {'xs': center_right_pred.squeeze(1)[:, 0:1], 'ys': center_right_pred.squeeze(1)[:, 1:2]}

        metrics = {
            "val_loss": loss,
            "val_p10_acc_L": p_acc(dets_left, center_left_gt, close_left, 10),
            "val_p5_acc_L": p_acc(dets_left, center_left_gt, close_left, 5),
            "val_p1_acc_L": p_acc(dets_left, center_left_gt, close_left, 1),
            "val_dist_L": cal_mean_distance(dets_left, center_left_gt, close_left),
            "val_p10_acc_R": p_acc(dets_right, center_right_gt, close_right, 10),
            "val_p5_acc_R": p_acc(dets_right, center_right_gt, close_right, 5),
            "val_p1_acc_R": p_acc(dets_right, center_right_gt, close_right, 1),
            "val_dist_R": cal_mean_distance(dets_right, center_right_gt, close_right),
        }
        if any(torch.isnan(v) for v in metrics.values()): return None

        self.validation_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        if not self.validation_outputs: print("No valid validation steps in this epoch."); return
        valid_results = [x for x in self.validation_outputs if x is not None]
        if not valid_results: print(
            "No valid metrics to aggregate in this epoch."); self.validation_outputs.clear(); return

        metrics = {key: torch.stack([x[key] for x in valid_results]).mean() for key in valid_results[0]}

        self.log("val_loss", metrics["val_loss"], prog_bar=True, sync_dist=True)
        self.log("val_p10_acc_L", metrics["val_p10_acc_L"], prog_bar=False, sync_dist=True);
        self.log("val_p5_acc_L", metrics["val_p5_acc_L"], prog_bar=False, sync_dist=True);
        self.log("val_p1_acc_L", metrics["val_p1_acc_L"], prog_bar=True, sync_dist=True);
        self.log("val_dist_L", metrics["val_dist_L"], prog_bar=True, sync_dist=True)
        self.log("val_p10_acc_R", metrics["val_p10_acc_R"], prog_bar=False, sync_dist=True);
        self.log("val_p5_acc_R", metrics["val_p5_acc_R"], prog_bar=False, sync_dist=True);
        self.log("val_p1_acc_R", metrics["val_p1_acc_R"], prog_bar=True, sync_dist=True);
        self.log("val_dist_R", metrics["val_dist_R"], prog_bar=True, sync_dist=True)

        print(f"\n====== Epoch {self.current_epoch} Validation Summary ======");
        print(f"  - val_loss    : {metrics['val_loss']:.4f}");
        print(f"  --- Left Eye ---");
        print(
            f"    - p10/p5/p1 Acc: {metrics['val_p10_acc_L']:.4f} / {metrics['val_p5_acc_L']:.4f} / {metrics['val_p1_acc_L']:.4f}");
        print(f"    - Dist         : {metrics['val_dist_L']:.4f}");
        print(f"  --- Right Eye ---");
        print(
            f"    - p10/p5/p1 Acc: {metrics['val_p10_acc_R']:.4f} / {metrics['val_p5_acc_R']:.4f} / {metrics['val_p1_acc_R']:.4f}");
        print(f"    - Dist         : {metrics['val_dist_R']:.4f}");
        print("=======================================\n")

        self.validation_outputs.clear()

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("learning_rate", 1e-3),
                                     weight_decay=self.hparams.get("weight_decay", 1e-5))
        scheduler = StepLRScheduler(optimizer, decay_t=self.hparams.get("decay_t", 10),
                                    decay_rate=self.hparams.get("decay_rate", 0.7),
                                    warmup_lr_init=self.hparams.get("warmup_lr_init", 1e-5),
                                    warmup_t=self.hparams.get("warmup_t", 5))
        return dict(optimizer=optimizer, lr_scheduler=scheduler)