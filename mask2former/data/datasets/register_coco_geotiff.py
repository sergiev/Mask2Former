# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances, load_sem_seg

COCO_CATEGORIES = [
    {
        'id': 1,
        'name': 'uchastok',
        'supercategory': 'land',
    },
] 

INSTANCE_SUBSETS = [
        ("uchastok_instance_train", "train/uchastok_2021", "train/uchastok_train2021.json"),
        ("uchastok_instance_test","test/uchastok_2021", "test/uchastok_test2021.json"),
        ("uchastok_instance_eval", "eval/uchastok_2021", "eval/uchastok_eval2021.json")
]

SEMANTIC_SUBSETS = [
        ("uchastok_semantic_train", "train/uchastok_2021", "train/anno/semantic"),
        ("uchastok_semantic_test","test/uchastok_2021", "test/anno/semantic"),
        ("uchastok_semantic_eval", "eval/uchastok_2021", "eval/anno/semantic")
]

PANOPTIC_SUBSETS = [
        ("uchastok_panoptic_train", "train/uchastok_2021", "train/uchastok_train2021.json"),
        ("uchastok_panoptic_test","test/uchastok_2021", "test/uchastok_test2021.json"),
        ("uchastok_panoptic_eval", "eval/uchastok_2021", "eval/uchastok_eval2021.json")
]

ROOT = '/mnt/localssd/MaskToFormer/data/dataset'

def _get_coco_stuff_meta():
    stuff_ids = [k["id"] for k in COCO_CATEGORIES]

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def register_semantic_segmentation(root=ROOT):
    meta = _get_coco_stuff_meta()
    for subset_name, image_dir, sem_seg_dir in SEMANTIC_SUBSETS:
        full_sem_seg_root = os.path.join(root, sem_seg_dir)
        full_image_root = os.path.join(root, image_dir)
        DatasetCatalog.register(subset_name, 
            lambda x=full_sem_seg_root, y=full_image_root: load_sem_seg(x, y, gt_ext='tif', image_ext='tif'))
        MetadataCatalog.get(subset_name).set(
            image_root=full_image_root,
            sem_seg_root=full_sem_seg_root,
            evaluator_type="sem_seg",
            ignore_label=0,
            **meta,
        )

def register_instance_segmentation(root=ROOT):
    for subset_name, image_dir, anno_name in INSTANCE_SUBSETS:
        full_image_root = os.path.join(root, image_dir)
        full_anno_path = os.path.join(root, anno_name)
        register_coco_instances(subset_name, {}, full_anno_path, full_image_root)

def register_panoptic_segmentation(root=ROOT):
    pass

register_semantic_segmentation()
register_instance_segmentation()
