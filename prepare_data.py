import os
from PIL import Image
import random
from tqdm import tqdm


def crop_resize(data_dir, save_dir, new_size=(1920, 1080)):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")
    
    new_image_dir = os.path.join(save_dir, "images")
    os.makedirs(new_image_dir)
    new_label_dir = os.path.join(save_dir, "labels")
    os.makedirs(new_label_dir)
    
    image_names = sorted(os.listdir(image_dir))
    for image_name in tqdm(image_names):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name[:-3] + "png")

        image = Image.open(image_path)
        label = Image.open(label_path)
        w, h = image.size
        new_h = w // 2

        if new_h <= h:
            h1 = random.randint((h - new_h) // 2, h - new_h)

            image = image.crop((0, h1, w, h1 + new_h))
            label = label.crop((0, h1, w, h1 + new_h))

        new_image = image.resize(new_size)
        new_label = label.resize(new_size)

        new_image_path = os.path.join(new_image_dir, image_name)
        new_label_path = os.path.join(new_label_dir, image_name[:-3] + "png")

        new_image.save(new_image_path)
        new_label.save(new_label_path)


def main():
    data_root = "/datasets/Mapillary/mapillary-vistas-dataset_public_v1.1"
    new_size = (1920, 1080)
    for mode in ["training", "validation"]:
        data_dir = os.path.join(data_root, mode)
        save_dir = os.path.join("Mapillary", mode)
        os.makedirs(save_dir, exist_ok=True)
        crop_resize(data_dir, save_dir, new_size)


if __name__ == "__main__":
    main()
