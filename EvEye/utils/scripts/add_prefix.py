import os


def add_prefix_to_png_files(directory, prefix="right_"):

    for filename in os.listdir(directory):

        if filename.endswith(".png"):

            old_file = os.path.join(directory, filename)

            new_filename = prefix + filename

            new_file = os.path.join(directory, new_filename)

            os.rename(old_file, new_file)
            print(f"Renamed {old_file} to {new_file}")


def main():
    target_directory = "/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/right/data"
    add_prefix_to_png_files(target_directory)


if __name__ == "__main__":
    main()
