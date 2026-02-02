from PIL import Image

def check_png_channels(file_path):
    """检查PNG图像的通道数量"""
    try:
        # 打开图像文件
        with Image.open(file_path) as img:
            # 确保图像是PNG格式
            if img.format != 'PNG':
                return f"错误：文件 {file_path} 不是PNG格式"
            
            # 获取图像模式和通道信息
            mode = img.mode
            channels = img.getbands()
            
            return {
                'file_path': file_path,
                'mode': mode,
                'channels': len(channels),
                'channel_names': channels
            }
    
    except FileNotFoundError:
        return f"错误：找不到文件 {file_path}"
    except Exception as e:
        return f"错误：处理文件时发生异常 - {str(e)}"

if __name__ == "__main__":
    # 直接在代码中修改文件路径
    file_path = "D:/EV-Eye-main/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/outframes/events/00031_1657710793649759.png"  # 请替换为实际的PNG文件路径
    
    result = check_png_channels(file_path)
    
    if isinstance(result, dict):
        print(f"文件: {result['file_path']}")
        print(f"图像模式: {result['mode']}")
        print(f"通道数量: {result['channels']}")
        print(f"通道名称: {', '.join(result['channel_names'])}")
    else:
        print(result)    