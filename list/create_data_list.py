import os


def get_filepaths(img_folder, mask_folder, filepath):
    img_filenames = sorted(os.listdir(img_folder))
    mask_filenames = sorted(os.listdir(mask_folder))
    with open(filepath, "a") as f:
        for img_filename, mask_filename in zip(img_filenames, mask_filenames):
            img_path = os.path.join(img_folder, img_filename)
            mask_path = os.path.join(mask_folder, mask_filename)
            f.write(f"{img_path};{mask_path}\n")


def main():
    test_only = False
    if test_only:
        folder = "/home/adeshkin/Desktop/seg_dyn_map/taganrog/imgs/2019-08-16-12-06-24_14_a_color"
        name = "taganrog_day"
        filepath = f"./test_{name}.txt"
        get_filepaths(folder, folder, filepath)
        return

    resolution = "1920_1080"
    for mode in ["train", "val"]:
        root = f"/media/cds-k/Data_2/DATASETS/Mapillary/{mode}/{resolution}"
        img_folder = os.path.join(root, "images")
        mask_folder = os.path.join(root, "labels")
        filepath = f"./{mode}_mapillary_{resolution}.txt"
        get_filepaths(img_folder, mask_folder, filepath)


if __name__ == "__main__":
    main()