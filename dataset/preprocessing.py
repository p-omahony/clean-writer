import pandas as pd
import os

from skimage import io, transform, color
import numpy as np

import torch
import torchvision.transforms as T

def read_labels(txt_file):
    gt = []
    for line in open(txt_file):
        if not line.startswith("#"):
            info = line.strip().split()
            if info[1] == 'ok':
                gt.append((info[0] + '.png', ' '.join(info[8:]).replace('|', ' ').lower()))
    df = pd.DataFrame(gt, columns=['file', 'word'])
    return df

def get_mapping(df):
    chars = []
    df.iloc[:, -1].apply(lambda x: chars.extend(list(x)))
    chars = sorted(list(set(chars)))
    char_dict = {c: i for i, c in enumerate(chars)}
    return char_dict

def resize_img(img, output_size, border_pad):
    resize = (output_size[0] - border_pad[0], output_size[1] - border_pad[1])
    h, w = img.shape[:2]
    fx = w / resize[1]
    fy = h / resize[0]
    f = max(fx, fy)
    new_size = (max(min(resize[0], int(h / f)), 1), max(min(resize[1], int(w / f * np.random.uniform(1, 3))), 1))
    image = transform.resize(img, new_size, preserve_range=True, mode='constant', cval=255)

    return image, new_size

def create_canvas(img, output_size, new_size):
    canvas = np.ones(output_size, dtype=np.uint8) * 255
    v_pad_max = output_size[0] - new_size[0]
    h_pad_max = output_size[1] - new_size[1]
    v_pad = int(np.random.choice(np.arange(0, v_pad_max + 1), 1))
    h_pad = int(np.random.choice(np.arange(0, h_pad_max + 1), 1))
    canvas[v_pad:v_pad + new_size[0], h_pad:h_pad + new_size[1]] = img

    return canvas


def transform_canvas(canvas):
    canvas = transform.rotate(canvas, -90, resize=True)[:, :-1]
    canvas = color.gray2rgb(canvas)
    canvas = torch.from_numpy(canvas.transpose((2, 0, 1))).float()
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return norm(canvas)

def preprocess_img(img_name, word):
    root_dir = './data'
    all_lines = ' '

    im_nm_split = img_name.split('-')
    start_folder = im_nm_split[0]
    src_folder = '-'.join(im_nm_split[:2])
    folder_name = os.path.join(start_folder, src_folder)
    img_filepath = os.path.join(root_dir, folder_name, img_name)

    all_lines = all_lines + ' ' + word
    border_pad = (4,10)
    output_size = (64, 800)

    image = io.imread(img_filepath)
    image, new_size = resize_img(image, output_size, border_pad)
    
    canvas = create_canvas(image, output_size, new_size)

    final_canvas = transform_canvas(canvas)

    return final_canvas, word
