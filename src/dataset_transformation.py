import os
import glob
import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import pandas as pd

def get_square_crop(mask):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    side_length = max(x_max - x_min, y_max - y_min)

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    x1 = max(0, center_x - side_length // 2)
    y1 = max(0, center_y - side_length // 2)
    x2 = min(mask.shape[1], x1 + side_length)
    y2 = min(mask.shape[0], y1 + side_length)

    if x1 == 0:
        x2 = side_length
    if y1 == 0:
        y2 = side_length
    if x2 == mask.shape[1]:
        x1 = x2 - side_length
    if y2 == mask.shape[0]:
        y1 = y2 - side_length

    return int(x1), int(y1), int(x2), int(y2)

def process_image (path_img, model, target_class_num, conf=0.25):
    img = cv2.imread(path_img)
    results = model(img, conf=conf)

    w,h = img.shape[1], img.shape[0]

    for result in results:
        masks = result.masks
        if masks is None:
            print(f"No segmentation mask found for image {path_img}")
            continue

        min_d = 1e6
        xyxy = None
        for i, mask in enumerate(masks):
            if result.boxes[i].cls == target_class_num:
                mask_array = mask.data.cpu().numpy()[0]
                mask_array = cv2.resize(mask_array, (img.shape[1], img.shape[0]))
                binary_mask = (mask_array > 0.5).astype(np.uint8) * 255

                x1, y1, x2, y2 = get_square_crop(binary_mask)
                xm = (x1 + x2) // 2
                ym = (y1 + y2) // 2
                dx = abs(w // 2 - xm)
                dy = abs(h // 2 - ym)
                d = dx + dy
                if d < min_d:
                    min_d = d
                    xyxy = (x1, y1, x2, y2)
        if xyxy is not None:
            x1, y1, x2, y2 = map(int, xyxy)  # Ensure integers

            # Clip coordinates to image boundaries
            height, width = img.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Check if extreme aspect ratio
            crop_width = x2 - x1
            crop_height = y2 - y1
            aspect_ratio = crop_width / crop_height if crop_height != 0 else float('inf')
            print(f"Clipped xyxy: ({x1}, {y1}, {x2}, {y2}), Aspect ratio: {aspect_ratio:.2f}")

            # Define thresholds for extreme aspect ratios
            min_aspect_ratio = 0.2  # Adjust this value as needed
            max_aspect_ratio = 12.0  # Adjust this value as needed

            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                img = img[y1:y2, x1:x2]
                return img
            else:
                print(f"Extreme aspect ratio detected: {aspect_ratio:.2f}. Skipping this crop.")
                return img
        else:
            print(f"No target class found for image {path_img}")
            return img

if __name__ == '__main__':
    ## check torch
    try:
        import torch
        print('torch.__version__', torch.__version__)
        print('torch.cuda.is_available()', torch.cuda.is_available())
        print('torch.cuda.current_device()', torch.cuda.current_device())
        print('torch.cuda.device_count()', torch.cuda.device_count())
        print('torch.cuda.get_device_name(0)', torch.cuda.get_device_name(0))
    except Exception as e:
        print(f"Error: {e}")

    try:
        import torchvision
        print('torchvision.__version__', torchvision.__version__)
    except Exception as e:
        print(f"Error: {e}")

    try:
        ## check ultralytics
        import ultralytics
        print('ultralytics.__version__', ultralytics.__version__)
    except Exception as e:
        print(f"Error: {e}")

    from ultralytics import YOLO

    path_model = os.path.join('..', 'models', 'yolov8', 'sunflowerstarornot.pt')
    model = YOLO(path_model)
    classes = model.names
    print(classes)
    target_class = 'Pycnopodia_helianthoides'
    print(f"Target class: {target_class}")
    target_class_num = None
    for key in classes.keys():
        class_name = classes[key]
        print(key, class_name)
        if class_name == target_class:
            target_class_num = key
            break
    print(f"Target class number: {target_class_num}")

    src_data_dir = os.path.join('..','archive', 'raw_sequence_sorted')
    if os.path.exists(src_data_dir) != True:
        raise Exception(f"Error: Source data directory does not exist: {src_data_dir}")

    dst_data_dir = os.path.join('..', 'archive', 'cropped_sequence_sorted')
    if os.path.exists(dst_data_dir) != True:
        os.makedirs(dst_data_dir)

    image_directories = glob.glob(os.path.join(src_data_dir, '*', '*'))
    image_directories = [d for d in image_directories if os.path.isdir(d)]
    ## reverse the order
    image_directories = image_directories[::-1]

    for image_dir in image_directories:
        print(f"Processing images in directory: {image_dir}")
        src_img_seq_dir = os.path.dirname(image_dir)
        dst_image_seq_dir = os.path.join(dst_data_dir, os.path.basename(src_img_seq_dir))
        if os.path.exists(dst_image_seq_dir) != True:
            os.makedirs(dst_image_seq_dir)
        dst_image_dir = os.path.join(dst_image_seq_dir, os.path.basename(image_dir))
        if os.path.exists(dst_image_dir) != True:
            os.makedirs(dst_image_dir)

        id_dst_img_dir = os.path.basename(dst_image_dir)
        print(f"ID image directory: {id_dst_img_dir}")
        path_best_csv = os.path.join(src_img_seq_dir, 'best_pics.csv')
        print(f"Reading best image from: {path_best_csv}")
        df = pd.read_csv(path_best_csv)
        df = df[df['ids'].astype(str) == str(id_dst_img_dir)]
        if df.empty:
            name_best_im = None
        else:
            name_best_im = df['path_to_best_im'].values[0]

        image_files = glob.glob(os.path.join(image_dir, '*.png')) + glob.glob(os.path.join(image_dir, '*.jpg'))
        ## sort the image files so that the best image is processed first
        if name_best_im is not None:
            image_files.sort(key=lambda x: os.path.basename(x) == name_best_im, reverse=True)
        print(f"Found {len(image_files)} image files in directory {image_dir}")
        print(f"Best image: {name_best_im}, processing from {image_files[0]}")

        pbar = tqdm(image_files, desc='Processing images', position=0, leave=True)
        for k, image_file in enumerate(image_files):
            print(f"Processing image: {image_file}")
            dst_image_file = os.path.join(dst_image_dir, str(k) + '__' + os.path.basename(image_file))
            if os.path.exists(dst_image_file):
                print(f"File exists: {dst_image_file}")
                pbar.update(1)
                continue

            img = process_image(image_file, model, target_class_num)
            if img is not None:
                cv2.imwrite(dst_image_file, img)
                print(f"Saved cropped image: {dst_image_file}")
            else:
                print(f"Error processing image: {image_file}")
            pbar.update(1)
        pbar.close()
    print("Processing completed")