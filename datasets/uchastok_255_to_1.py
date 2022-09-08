#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
import cv2

in_pattern='/mnt/localssd/MaskToFormer/data/dataset/{}/annotations_semantic'
out_pattern='/mnt/localssd/MaskToFormer/data/dataset/{}/anno/semantic'
subsets = ['train', 'test', 'eval']

def convert(input_path, output_path):
    img = cv2.imread(input_path)
    assert img.dtype == np.uint8
    img[img==255] = 1  # 0 (ignore) becomes 255. others are shifted by 1
    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "ADEChallengeData2016"
    for subset in subsets:
        input_dir = in_pattern.format(subset)
        output_dir = out_pattern.format(subset)
        os.makedirs(output_dir, exist_ok=True)
        for file_name in tqdm.tqdm(os.listdir(input_dir)):
            input_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)
            convert(input_file, output_file)
