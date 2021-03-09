import os
import cv2
import datetime
import time
import argparse
import transforms as T

from vis_utils.config import METADATA_STUFF_CLASSES
from vis_utils.config import METADATA_STUFF_COLORS

from vis_utils.visualizer import Visualizer
from vis_utils.color_mode import ColorMode


def parse_args():
    parser = argparse.ArgumentParser(description='Visualization of semantic segmentation masks with class names')
    
    parser.add_argument('--image-dir', metavar='DIR', default=None, help='path to images')
    parser.add_argument('--mask-dir', metavar='DIR', default=None, help='path to masks')
    parser.add_argument('--save-dir', default=None, help='path where to put amazing outputs')

    args = parser.parse_args()
    
    return args


def vis_one_image(image, mask, save_name, save_dir):  
    # Convert image from OpenCV BGR format to Matplotlib RGB format.
    #image = image[:, :, ::-1]
        
    visualizer = Visualizer(image, METADATA_STUFF_CLASSES, METADATA_STUFF_COLORS, instance_mode=ColorMode.SEGMENTATION)
    vis_output = visualizer.draw_sem_seg(mask)
    
    save_path = os.path.join(save_dir, save_name)
    vis_output.save(save_path)

    
def vis_from_directory(image_dir, mask_dir, save_dir, image_paths, mask_paths):
    for i, (image_name, mask_name) in enumerate(zip(image_paths, mask_paths)):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        
        vis_one_image(image, mask, image_name, save_dir)
        
        if i % 10 == 9:
            print(f'{i} / {len(image_paths)} : {image_name}')


def get_paths(file_dir):
    paths = sorted([x for x in os.listdir(file_dir) if os.path.isfile(f'{file_dir}/{x}')])
    
    return paths


def get_transform():
    transforms_ = []
    transforms_.append(T.ToTensor())
    
    transforms_.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms_)


def main(args):
    os.makedirs(args.save_dir)
        
    image_paths = get_paths(args.image_dir)
    mask_paths = get_paths(args.mask_dir)

    print(f"Visualization:\n image_dir: {args.image_dir}")
    print(f" mask_dir: {args.mask_dir}")

    start_time = time.time()

    vis_from_directory(args.image_dir, args.mask_dir, args.save_dir, image_paths, mask_paths)
    
    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time))) 
    print('Visualization time {}'.format(total_time_str)) 
    print(f'save_dir: {args.save_dir}')


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
