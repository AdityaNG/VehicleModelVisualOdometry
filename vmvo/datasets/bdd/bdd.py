import glob
import os

import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from deviant.lib.datasets.kitti_utils import (
    get_affine_transform,
)  # Calibration,; affine_transform,; compute_box_3d,; get_objects_from_label,

from .bdd_raw import AndroidDatasetIterator
from .helper import DATASET_LIST, ROOT_DATASET_DIR


class BDDPrimitive:
    def __init__(
        self,
        cfg,
    ):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ["Pedestrian", "Car", "Cyclist"]
        self.cls2id = {"Pedestrian": 0, "Car": 1, "Cyclist": 2}
        self.resolution = (
            np.array([1280, 384])
            if "resolution" not in cfg.keys()
            else np.array(cfg["resolution"])
        )  # W * H
        self.eval_dataset = (
            cfg["type"]
            if "eval_dataset" not in cfg.keys()
            else cfg["eval_dataset"]
        )
        self.use_3d_center = cfg["use_3d_center"]
        self.writelist = cfg["writelist"]
        if cfg["class_merging"]:
            self.writelist.extend(["Van", "Truck"])
        if cfg["use_dontcare"]:
            self.writelist.extend(["DontCare"])
        # l,w,h
        self.cls_mean_size = np.array(
            [
                [1.76255119, 0.66068622, 0.84422524],
                [1.52563191462, 1.62856739989, 3.88311640418],
                [1.73698127, 0.59706367, 1.76282397],
            ]
        )

        # # data split loading
        # assert (
        #     split in ["train", "val", "train2", "val2", "trainval", "test"]
        #     or "train" in split
        # )
        # self.split = split

        # data augmentation configuration
        self.data_augmentation = False
        self.random_flip = cfg["random_flip"]
        self.random_crop = cfg["random_crop"]
        self.scale = cfg["scale"]
        self.shift = cfg["shift"]

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4


class BenagaluruBoundingBoxDataset(BDDPrimitive):
    def __init__(
        self,
        **kwargs,
    ):
        cfg = kwargs["cfg"]
        super().__init__(cfg)
        # Remove cfg
        del kwargs["cfg"]
        self.dataset = AndroidDatasetIterator(**kwargs)

        # bounding box labels
        self.bbox_path = os.path.join(self.dataset.folder_path, "bbox_labels")

        # List of all the available bounding box labels
        self.bbox_labels = sorted(os.listdir(self.bbox_path))
        # Remove *.npy from the name
        self.bbox_labels = [x[:-4] for x in self.bbox_labels]

        # List of all the available images
        print(self.bbox_labels)

    def __len__(self):
        return len(self.bbox_labels)

    def __getitem__(self, index):
        timestamp = int(self.bbox_labels[index])
        adjusted_index = self.dataset.csv_dat["Timestamp"].searchsorted(
            timestamp
        )

        return self.dataset[adjusted_index]


class BDD(Dataset):
    def __init__(
        self,
        cfg: dict,
        folder_path: str = DATASET_LIST[-1],
    ):
        self.dataset = BenagaluruBoundingBoxDataset(
            folder_path=folder_path,
            compute_trajectory=True,  # Load the VO trajectory
            invalidate_cache=False,  # Do not clear cache
            cfg=cfg,
        )
        self.cfg = cfg
        self.P2 = np.array(
            [
                [1250.6, 0.000000e00, 978.4, 0.000000e00],
                [0.000000e00, 1254.8, 562.1, 0.000000e00],
                [0.000000e00, 0.000000e00, 1.000000e00, 0.000000e00],
            ]
        )
        self.data_augmentation = cfg["data_augmentation"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # csv_frame, img_frame = self.dataset[index]
        _, img_frame = self.dataset[index]
        # timestamp = int(csv_frame["Timestamp"])

        #  ============================   get inputs   ========================
        # Convert image from cv2 to PIL
        img = Image.fromarray(img_frame)
        img_size = np.array(img.size)

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        # random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.dataset.random_flip:
                # random_flip_flag = True
                img = img.transpose(
                    Image.FLIP_LEFT_RIGHT  # pylint: disable=E1101
                )

            if np.random.random() < self.dataset.random_crop:
                # random_crop_flag = True
                crop_size = img_size * np.clip(
                    np.random.randn() * self.dataset.scale + 1,
                    1 - self.dataset.scale,
                    1 + self.dataset.scale,
                )
                center[0] += img_size[0] * np.clip(
                    np.random.randn() * self.dataset.shift,
                    -2 * self.dataset.shift,
                    2 * self.dataset.shift,
                )
                center[1] += img_size[1] * np.clip(
                    np.random.randn() * self.dataset.shift,
                    -2 * self.dataset.shift,
                    2 * self.dataset.shift,
                )

        # add affine transformation for 2d images.
        # trans, trans_inv = get_affine_transform(
        _, trans_inv = get_affine_transform(
            center, crop_size, 0, self.dataset.resolution, inv=1
        )
        img = img.transform(
            tuple(self.dataset.resolution.tolist()),
            method=Image.AFFINE,  # pylint: disable=E1101
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.Resampling.BILINEAR,
        )
        coord_range = np.array(
            [center - crop_size / 2, center + crop_size / 2]
        ).astype(np.float32)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.dataset.mean) / self.dataset.std
        img = img.transpose(2, 0, 1)  # C * H * W

        # calib = self.get_calib(index)
        features_size = (
            self.dataset.resolution // self.dataset.downsample
        )  # W * H
        #  ============================   get labels   ========================
        heatmap = np.zeros(
            (self.dataset.num_classes, features_size[1], features_size[0]),
            dtype=np.float32,
        )  # C * H * W
        size_2d = np.zeros((self.dataset.max_objs, 2), dtype=np.float32)
        offset_2d = np.zeros((self.dataset.max_objs, 2), dtype=np.float32)
        depth = np.zeros((self.dataset.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.dataset.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.dataset.max_objs, 1), dtype=np.float32)
        # src_size_3d = np.zeros((self.dataset.max_objs, 3), dtype=np.float32)
        size_3d = np.zeros((self.dataset.max_objs, 3), dtype=np.float32)
        offset_3d = np.zeros((self.dataset.max_objs, 2), dtype=np.float32)
        # height2d = np.zeros((self.dataset.max_objs, 1), dtype=np.float32)
        cls_ids = np.zeros((self.dataset.max_objs), dtype=np.int64)
        indices = np.zeros((self.dataset.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.dataset.max_objs), dtype=np.uint8)
        # mask_3d = np.zeros((self.dataset.max_objs), dtype=np.uint8)
        # targets = {}
        targets = {
            "depth": depth,
            "size_2d": size_2d,
            "heatmap": heatmap,
            "offset_2d": offset_2d,
            "indices": indices,
            "size_3d": size_3d,
            "offset_3d": offset_3d,
            "heading_bin": heading_bin,
            "heading_res": heading_res,
            "cls_ids": cls_ids,
            "mask_2d": mask_2d,
        }
        inputs = img
        info = {
            "img_id": index,
            "img_size": img_size,
            "bbox_downsample_ratio": img_size / features_size,
        }
        return inputs, self.P2, coord_range, targets, info


def get_all_bdd_datasets(
    iterator_class=BDD, root_dir=ROOT_DATASET_DIR, dataset_args=None
):
    if dataset_args is None:
        dataset_args = {}
    DATASET_DIR = os.path.join(root_dir, "android")
    DATASET_LIST_LOCAL = sorted(glob.glob(DATASET_DIR + "/*"))

    # Exclude dataset/android/calibration
    DATASET_LIST_LOCAL = [
        dataset
        for dataset in DATASET_LIST_LOCAL
        if "calibration" not in dataset
    ]

    DAYTIME_IDS = [
        # "1653972957447",
        # "1652937970859",
        "1658384707877",
        # "1658384924059",
        # "1654507149598",
        # "1654493684259",
    ]

    DAYTIME_LIST = [
        dataset
        for dataset in DATASET_LIST_LOCAL
        if dataset.split("/")[-1] in DAYTIME_IDS
    ]

    return get_bdd_datasets(
        iterator_class=iterator_class,
        folders_list=DAYTIME_LIST,
        # folders_list=DATASET_LIST_LOCAL,
        dataset_args=dataset_args,
    )


def get_bdd_datasets(
    iterator_class=BDD, folders_list=DATASET_LIST, dataset_args=None
):
    if dataset_args is None:
        dataset_args = {}
    datasets_list = []
    for root_dir in folders_list:
        dataset = iterator_class(folder_path=root_dir, **dataset_args)
        datasets_list.append(dataset)
    datasets = ConcatDataset(datasets_list)
    return datasets, datasets_list
