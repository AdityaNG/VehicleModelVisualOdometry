import glob
import os
from pathlib import Path

# ROOT_DATASET_DIR = "dataset"
ROOT_DATASET_DIR = os.path.join(
    str(Path(__file__).parent.parent.absolute()), "dataset"
)
ROOT_DATASET_DIR = os.path.join(os.path.expanduser("~/Datasets"), "dataset")

DATASET_DIR = os.path.join(ROOT_DATASET_DIR, "android")
DATASET_LIST = sorted(glob.glob(DATASET_DIR + "/*"))

# Exclude dataset/android/calibration
DATASET_LIST = [
    dataset for dataset in DATASET_LIST if "calibration" not in dataset
]

if len(DATASET_LIST) == 0:
    DATASET_LIST = ["dataset/android/"]
TRAJECTORY_CACHE_DIR = ".trajectory_cache"

SETTINGS_DOC = os.path.expanduser(
    "~/Datasets/Depth_Dataset_Bengaluru/calibration/pocoX3/calib.yaml"
)

DAYTIME_IDS = [
    "1653972957447",
    "1652937970859",
    "1658384707877",
    "1658384924059",
    "1654507149598",
    "1654493684259",
]

NIGHTTIME_IDS = [
    "1654007317545",
    "1654006195191",
    "1654005945981",
]

DAYTIME_LIST = [
    dataset
    for dataset in DATASET_LIST
    if dataset.split("/")[-1] in DAYTIME_IDS
]
NIGHTTIME_LIST = [
    dataset
    for dataset in DATASET_LIST
    if dataset.split("/")[-1] in NIGHTTIME_IDS
]
