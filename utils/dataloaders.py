# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Dataloaders and dataset utils."""

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
)
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    check_dataset,
    check_requirements,
    check_yaml,
    clean_str,
    cv2,
    is_colab,
    is_kaggle,
    segments2boxes,
    unzip_file,
    xyn2xy,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from utils.torch_utils import torch_distributed_zero_first
from utils.general import xyxy2xywhn
from .augmentations import random_perspective

# Parameters
HELP_URL = "See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(paths):
    """Generates a single SHA256 hash for a list of file or directory paths by combining their sizes and paths."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns corrected PIL image size (width, height) considering EXIF orientation."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def seed_worker(worker_id):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Inherit from DistributedSampler and override iterator
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
class SmartDistributedSampler(distributed.DistributedSampler):
    def __iter__(self):
        """Yields indices for distributed data sampling, shuffled deterministically based on epoch and seed."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # determine the eventual size (n) of self.indices (DDP indices)
        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1  # num_replicas == WORLD_SIZE
        idx = torch.randperm(n, generator=g)
        if not self.shuffle:
            idx = idx.sort()[0]

        idx = idx.tolist()
        if self.drop_last:
            idx = idx[: self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx)))[:padding_size]

        return iter(idx)


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    seed=0,
    rgbt_input=False,
):
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False

    dataset_class = LoadImagesAndLabels if not rgbt_input else LoadRGBTImagesAndLabels
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = dataset_class(
            path,
            img_size=imgsz,
            batch_size=batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            rank=rank,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=dataset_class.collate_fn4 if quad else dataset_class.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)


class LoadScreenshots:
    # YOLOv5 screenshot dataloader, i.e. `python detect.py --source "screen 0 100 100 512 256"`
    def __init__(self, source, img_size=640, stride=32, auto=True, transforms=None):
        """
        Initializes a screenshot dataloader for YOLOv5 with specified source region, image size, stride, auto, and
        transforms.

        Source = [screen_number left top width height] (pixels)
        """
        check_requirements("mss")
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = "stream"
        self.frame = 0
        self.sct = mss.mss()

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        """Iterates over itself, enabling use in loops and iterable contexts."""
        return self

    def __next__(self):
        """Captures and returns the next screen frame as a BGR numpy array, cropping to only the first three channels
        from BGRA.
        """
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s


class LoadImages:
    """YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`"""

    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initializes YOLOv5 loader for images/videos, supporting glob patterns, directories, and lists of paths."""
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        """Initializes iterator by resetting count and returns the iterator object itself."""
        self.count = 0
        return self

    def __next__(self):
        """Advances to the next file in the dataset, raising StopIteration if at the end."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """Initializes a new video capture object with path, frame count adjusted by stride, and orientation
        metadata.
        """
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        """Rotates a cv2 image based on its orientation; supports 0, 90, and 180 degrees rotations."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.nf  # number of files


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources="file.streams", img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initializes a stream loader for processing video streams with YOLOv5, supporting various sources including
        YouTube.
        """
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "
            if urlparse(s).hostname in ("www.youtube.com", "youtube.com", "youtu.be"):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
                check_requirements(("pafy", "youtube_dl==2020.12.2"))
                import pafy

                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), "--source 0 webcam unsupported on Colab. Rerun command in a local environment."
                assert not is_kaggle(), "--source 0 webcam unsupported on Kaggle. Rerun command in a local environment."
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"{st}Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")  # newline

        # check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning("WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.")

    def update(self, i, cap, stream):
        """Reads frames from stream `i`, updating imgs array; handles stream reopening on signal loss."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        """Resets and returns the iterator for iterating over video frames or images in a dataset."""
        self.count = -1
        return self

    def __next__(self):
        """Iterates over video frames or images, halting on thread stop or 'q' key press, raising `StopIteration` when
        done.
        """
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ""

    def __len__(self):
        """Returns the number of sources in the dataset, supporting up to 32 streams at 30 FPS over 30 years."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    """Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and
    extension with `.txt`.
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        min_items=0,
        prefix="",
        rank=-1,
        seed=0,
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}\n{HELP_URL}") from e

        # Check cache
        self.label_files = self.img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        self.labels, self.shapes, self.segments = [], [], []

        for ii, (im_file, label, shape, segment) in enumerate(cache['data']):
            assert im_file == self.im_files[ii]
            self.labels.append(label)
            self.shapes.append(shape)
            self.segments.append(segment)

        self.shapes = np.array(self.shapes)
        nl = len(np.concatenate(self.labels, 0))  # number of labels
        assert nl > 0 or not augment, f"{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        self.label_files = self.img2label_paths(self.im_files)

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f"{prefix}{n - len(include)}/{n} images filtered from dataset")
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)
        if rank > -1:  # DDP indices (see: SmartDistributedSampler)
            # force each rank (i.e. GPU process) to sample the same subset of data on every epoch
            self.indices = self.indices[np.random.RandomState(seed=seed).permutation(n) % WORLD_SIZE == RANK]

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        self.prepare_image_cache(cache_images, prefix, n)

    def prepare_image_cache(self, cache_images, prefix, n):
        # Cache images into RAM/disk for faster training
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            results = ThreadPool(NUM_THREADS).imap(lambda i: (i, fcn(i)), self.indices)
            pbar = tqdm(results, total=len(self.indices), bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes * WORLD_SIZE
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"
            pbar.close()

    def check_cache_ram(self, safety_margin=0.1, prefix=""):
        """Checks if available RAM is sufficient for caching images, adjusting for a safety margin."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Caches dataset labels, verifies images, reads shapes, and tracks dataset integrity."""
        x = {}  # dict
        data = []  # use list to keep image orders
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."

        pbar = tqdm(
            zip(self.im_files, self.label_files, repeat(prefix)),
            desc=desc,
            total=len(self.im_files),
            bar_format=TQDM_BAR_FORMAT)

        for blob in pbar:
            im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg = verify_image_label(blob)
            pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if msg:
                msgs.append(msg)
            data.append([im_file, lb, shape, segments])
        pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["data"] = data
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.im_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        
        shapes = None  # shapes는 non-mosaic 경로에서 설정되므로 초기화
        imgs_to_return = [] # 최종 반환될 이미지 텐서 리스트
        
        if mosaic:
            # Load mosaic for RGBT
            # self.load_mosaic는 ( (lwir_np, vis_np), labels_np_Nx6 )를 반환합니다.
            # labels_np_Nx6: [cls, xc_norm, yc_norm, w_norm, h_norm, occ_level]
            imgs_mosaic_np_tuple, labels_mosaic_np = self.load_mosaic(index) 
            
            if random.random() < hyp.get("mixup", 0.0):
                imgs2_mosaic_np_tuple, labels2_mosaic_np = self.load_mosaic(random.choice(self.indices))
                # 사용자 정의 RGBT mixup 함수 사용
                imgs_mosaic_np_tuple, labels_mosaic_np = mixup(
                    imgs_mosaic_np_tuple, labels_mosaic_np, imgs2_mosaic_np_tuple, labels2_mosaic_np
                )

            nl = len(labels_mosaic_np)
            # labels_out: [batch_idx_placeholder, cls, xc, yc, w, h, occ_level]
            labels_out = torch.zeros((nl, 7)) 
            if nl:
                labels_out[:, 1:] = torch.from_numpy(labels_mosaic_np)
            
            # 모자이크 NumPy 이미지들을 Tensor로 변환
            for img_modality_np in imgs_mosaic_np_tuple: # (lwir_np, vis_np)
                img_tensor = img_modality_np.transpose((2, 0, 1))  # HWC to CHW
                if img_tensor.shape[0] == 3:  # 3채널 이미지 (BGR -> RGB 가정)
                    img_tensor = img_tensor[::-1, :, :] 
                img_tensor = np.ascontiguousarray(img_tensor)
                imgs_to_return.append(torch.from_numpy(img_tensor))

        else: # Not mosaic
            imgs_modalities_np, hw0s_orig_list, hw_resized_list = self.load_image(index) 
            
            # non-mosaic 경로에서 최종 레이블 (증강 적용 후, NumPy 형태)
            # self.labels[index]는 [cls, xc, yc, w, h, occ] 형태의 정규화된 NumPy 배열
            labels_processed_np_final = self.labels[index].copy() 
                                                            
            M_random_perspective = None # random_perspective 변환 행렬 공유용

            temp_processed_imgs_tensors = [] # 각 모달리티의 최종 처리된 텐서를 임시 저장

            for ii, (img_np, hw0_tuple, hw_resized_tuple) in enumerate(zip(imgs_modalities_np, hw0s_orig_list, hw_resized_list)):
                h0_orig, w0_orig = hw0_tuple # 원본 높이, 너비
                # h_resized, w_resized = hw_resized_tuple # letterbox 이전 리사이즈된 크기

                target_shape_for_letterbox = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
                img_lb_np, ratio_tuple, pad_tuple = letterbox(img_np.copy(), target_shape_for_letterbox, auto=False, scaleup=self.augment)
                
                if ii == 0: 
                    shapes = (h0_orig, w0_orig), (ratio_tuple, pad_tuple) # COCO mAP 리스케일링용 정보

                current_labels_np_norm = self.labels[index].copy() # [cls, xc, yc, w, h, occ]
                
                # 레이블 좌표 변환: normalized xywh -> pixel xyxy (letterboxed 이미지 기준)
                labels_pixel_xyxy_on_lb = np.zeros((len(current_labels_np_norm), 5)) # cls, x1,y1,x2,y2
                if current_labels_np_norm.size > 0:
                    labels_pixel_xyxy_on_lb[:, 0] = current_labels_np_norm[:, 0]
                    # xywhn2xyxy는 정규화된 xc,yc,w,h를 입력받아 pixel xyxy로 변환
                    # 이 때 사용되는 w, h는 정규화의 기준이 되었던 이미지 크기 (letterbox 적용 전, 패딩 추가 전의 이미지 크기)
                    # 즉, ratio_tuple[0] * w0_orig, ratio_tuple[1] * h0_orig 가 됩니다.
                    # 혹은 letterbox 함수 내부의 new_unpad 크기입니다.
                    w_unpadded_resized = int(round(w0_orig * ratio_tuple[0]))
                    h_unpadded_resized = int(round(h0_orig * ratio_tuple[1]))

                    labels_pixel_xyxy_on_lb[:, 1:5] = xywhn2xyxy(
                        current_labels_np_norm[:, 1:5].copy(), # xc, yc, w, h 부분만 전달
                        w=w_unpadded_resized, 
                        h=h_unpadded_resized, 
                        padw=pad_tuple[0], 
                        padh=pad_tuple[1]
                    )
                
                img_to_augment_np = img_lb_np # 증강은 letterbox된 이미지에 적용
                labels_for_augment_xyxy = labels_pixel_xyxy_on_lb # pixel xyxy 형태

                if self.augment:
                    # random_perspective 적용 (labels_for_augment_xyxy는 pixel xyxy)
                    current_segments = self.segments[index].copy() if self.segments and self.segments[index] else []
                    # segments도 pixel 좌표로 변환 필요 (random_perspective는 pixel 좌표를 가정)
                    if current_segments and labels_for_augment_xyxy.size > 0 : # 세그먼트가 있고 레이블도 있을 때
                        # xyn2xy 함수는 정규화된 segment 좌표를 pixel 좌표로 변환
                        # 이 때 사용되는 w, h도 위와 동일하게 w_unpadded_resized, h_unpadded_resized 사용
                        segments_pixel = [xyn2xy(seg, w=w_unpadded_resized, h=h_unpadded_resized, padw=pad_tuple[0], padh=pad_tuple[1]) for seg in current_segments]
                    else:
                        segments_pixel = []

                    if ii == 0: 
                        img_to_augment_np, labels_for_augment_xyxy, M_random_perspective = random_perspective(
                            img_to_augment_np,
                            labels_for_augment_xyxy, # [cls, x1,y1,x2,y2]
                            segments_pixel,
                            degrees=hyp.get("degrees", 0.0), translate=hyp.get("translate", 0.0),
                            scale=hyp.get("scale", 0.0), shear=hyp.get("shear", 0.0),
                            perspective=hyp.get("perspective", 0.0)
                        )
                    else: 
                        img_to_augment_np, _, _ = random_perspective(img_to_augment_np, M=M_random_perspective)
                        # labels_for_augment_xyxy는 첫 번째 모달리티 기준으로 이미 변환됨

                    # Albumentations (pixel xyxy 레이블을 YOLO 포맷으로 변환하여 전달)
                    if self.albumentations and labels_for_augment_xyxy.size > 0:
                        h_img_for_alb, w_img_for_alb = img_to_augment_np.shape[:2]
                        # labels_for_augment_xyxy ([cls, x1,y1,x2,y2]) -> YOLO 포맷 ([xc_norm, yc_norm, w_norm, h_norm])
                        labels_yolo_for_alb = np.zeros((len(labels_for_augment_xyxy), 5))
                        labels_yolo_for_alb[:,0] = labels_for_augment_xyxy[:,0]
                        labels_yolo_for_alb[:,1:] = xyxy2xywhn(labels_for_augment_xyxy[:,1:5].copy(), w=w_img_for_alb, h=h_img_for_alb, clip=True, eps=1e-3)

                        transformed = self.albumentations( # Albumentations 클래스는 YOLO 포맷을 받는다고 가정
                            image=img_to_augment_np, 
                            bboxes=labels_yolo_for_alb[:, 1:].tolist(),
                            class_labels=labels_yolo_for_alb[:, 0].tolist()
                        )
                        img_to_augment_np = transformed['image']
                        if ii == 0: # 레이블은 첫 번째 모달리티 기준으로만 업데이트
                            labels_from_alb_yolo = np.array(transformed['bboxes'])
                            if labels_from_alb_yolo.size > 0:
                                # Albumentations 반환 (YOLO 포맷) -> pixel xyxy로 다시 변환하여 labels_for_augment_xyxy 업데이트
                                labels_for_augment_xyxy[:,1:5] = xywhn2xyxy(labels_from_alb_yolo, w=img_to_augment_np.shape[1], h=img_to_augment_np.shape[0])
                            else:
                                labels_for_augment_xyxy = np.array([])


                    # HSV (Visible 이미지, 즉 ii == 1 일 때만 적용 가정)
                    if ii == 1: 
                        augment_hsv(img_to_augment_np, hgain=hyp.get("hsv_h",0.0), sgain=hyp.get("hsv_s",0.0), vgain=hyp.get("hsv_v",0.0))

                    # Flip (이미지에 적용 후, 레이블은 ii == 0 일 때만 업데이트)
                    if random.random() < hyp.get("flipud",0.0):
                        img_to_augment_np = np.flipud(img_to_augment_np)
                        if ii == 0 and labels_for_augment_xyxy.size > 0:
                            labels_for_augment_xyxy[:, [2, 4]] = img_to_augment_np.shape[0] - labels_for_augment_xyxy[:, [4, 2]] 

                    if random.random() < hyp.get("fliplr",0.0):
                        img_to_augment_np = np.fliplr(img_to_augment_np)
                        if ii == 0 and labels_for_augment_xyxy.size > 0:
                            labels_for_augment_xyxy[:, [1, 3]] = img_to_augment_np.shape[1] - labels_for_augment_xyxy[:, [3, 1]] 
                    
                if ii == 0: # 최종 증강된 레이블(pixel xyxy)을 labels_processed_final_np에 저장
                    labels_processed_final_np = labels_for_augment_xyxy # [cls, x1,y1,x2,y2]

                # 이미지 NumPy 배열을 Tensor로 변환
                img_tensor = img_to_augment_np.transpose((2, 0, 1))  # HWC to CHW
                if img_tensor.shape[0] == 3: # 3채널 (BGR -> RGB)
                    img_tensor = img_tensor[::-1, :, :]
                img_tensor = np.ascontiguousarray(img_tensor)
                temp_processed_imgs_tensors.append(torch.from_numpy(img_tensor))
            
            # non-mosaic 경로의 최종 레이블 처리 (pixel xyxy -> normalized xywh 및 occlusion 포함하여 labels_out 생성)
            # ---------- MOSAIC ----------
            nl = len(labels_mosaic_np)
            labels_out = torch.zeros((nl, 7))
            if nl:
                labels_out[:, 1:] = torch.from_numpy(labels_mosaic_np)   # cls, xc, yc, w, h, occ
            # -----------------------------------------------------------


            # ---------- NON-MOSAIC ----------
            nl = len(labels_processed_final_np)
            labels_out = torch.zeros((nl, 7))
            if nl:
                # cls
                labels_out[:, 1] = torch.from_numpy(labels_processed_final_np[:, 0])

                # pixel xyxy ➜ normalised xywh (최종 이미지 크기 기준)
                h_f, w_f = temp_processed_imgs_tensors[0].shape[1:3]
                labels_out[:, 2:6] = torch.from_numpy(
                    xyxy2xywhn(labels_processed_final_np[:, 1:5].copy(),
                            w=w_f, h=h_f, clip=True, eps=1e-3)
                )

                # occlusion
                if self.labels[index].shape[1] == 6 and len(self.labels[index]) == nl:
                    labels_out[:, 6] = torch.from_numpy(self.labels[index][:, 5])

            imgs_to_return = temp_processed_imgs_tensors


        # 공통 로직: labels_out의 마지막 컬럼 (occlusion level) 제거하여 최종 (N, 6) 형태로 만듦
        # labels_out은 이 시점에서 (N, 7) [batch_idx_placeholder, cls, xc, yc, w, h, occ] 형태를 가집니다.
        if labels_out.shape[1] == 7: # 7개의 컬럼을 가지고 있다면
             labels_out = labels_out[:, :-1] # 마지막 occlusion level 컬럼 제거 -> (N, 6)
        elif len(labels_out) > 0 and labels_out.shape[1] != 6: # 레이블이 있는데 컬럼 수가 6이 아니면 경고
            LOGGER.warning(f"labels_out has unexpected shape {labels_out.shape} before final processing for {self.im_files[index]}. Expected 6 or 7 columns.")
        # 만약 nl == 0 이면 labels_out은 (0,7) 또는 (0,6) 이고, 슬라이싱은 문제 없음.

        return imgs_to_return, labels_out, self.im_files[index], shapes, index

    def load_image(self, i):
        """
        Loads an image by index, returning the image, its original dimensions, and resized dimensions.

        Returns (im, original hw, resized hw)
        """
        im, f, fn = (
            self.ims[i],
            self.im_files[i],
            self.npy_files[i],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        """Saves an image to disk as an *.npy file for quicker loading, identified by index `i`."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        """Loads a 4-image mosaic for YOLOv5, combining 1 selected and 3 random images, with labels and segments."""
        labels4, segments4 = [], []
        s = self.img_size
        border_x, border_y = self.mosaic_border
        xc = int(random.uniform(-border_x, 2 * s + border_x))
        yc = int(random.uniform(-border_y, 2 * s + border_y))  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4

    def load_mosaic9(self, index):
        """Loads 1 image + 8 random images into a 9-image mosaic for augmented YOLOv5 training, returning labels and
        segments.
        """
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady :, x1 - padx :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc : yc + 2 * s, xc : xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp["copy_paste"])
        img9, labels9 = random_perspective(
            img9,
            labels9,
            segments9,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img9, labels9

    def img2label_paths(self, img_paths):
        """For compatibility with multispectral data"""
        return img2label_paths(img_paths)

    @staticmethod
    def collate_fn(batch):
        """Batches images, labels, paths, shapes, and indices assigning unique indices to targets in merged label tensor."""
        im, label, path, shapes, indices = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, indices

    @staticmethod
    def collate_fn4(batch):
        """Bundles a batch's data by quartering the number of shapes and paths, preparing it for model input."""
        im, label, path, shapes, indices = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4, indices4 = [], [], path[:n], shapes[:n], indices[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False)[
                    0
                ].type(im[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im1)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4, indices4


class LoadRGBTImagesAndLabels(LoadImagesAndLabels):
    cache_version = 0.2  # dataset labels *.cache version
    modalities = ('lwir', 'visible')
    ignore_settings = {
        'train': {  # standard training setting for KAIST dataset
            'hRng':( 12/512,  np.inf),
            'wRng':(-np.inf,  np.inf),
            'xRng':(  5/640, 1-5/640),
            'yRng':(  5/512, 1-5/512),
            },
        'test': {
            'hRng':(-np.inf, np.inf),
            'wRng':(-np.inf, np.inf),
            'xRng':(  5/640, 1-5/640),
            'yRng':(  5/512, 1-5/512),
        }
    }

    def __init__(self, path, **kwargs):
        # HACK: cannot guarantee that path contain split name
        is_train = 'train' in path
        single_cls = kwargs['single_cls']
        kwargs['single_cls'] = False
        assert kwargs['cache_images'] != 'ram', 'Image caching for RGBT dataset is not implemented yet.'
        if not is_train:
            assert not kwargs['rect'], 'Please do not turn-on "rect" option for validation. ' \
                                  'It causes shuffling of images and breaks the kaist evaluation pipeline.'

        super().__init__(path, **kwargs)

        # TODO: make mosaic augmentation work
        self.mosaic = self.augment and not self.rect

        # Set ignore flag
        cond = self.ignore_settings['train' if is_train else 'test']
        for i in range(len(self.labels)):
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][self.labels[i][:, 0] != 0, 0] = -1   # ignore cyclist / people / person

            if len(self.labels[i]):
                x1, y1, w, h = self.labels[i][:,1:5].T
                x2 = x1 + w
                y2 = y1 + h
                ignore_idx = (x1 < cond['xRng'][0]) & \
                            (x2 > cond['xRng'][1]) & \
                            (y1 < cond['yRng'][0]) & \
                            (y2 > cond['yRng'][1]) & \
                            (w < cond['wRng'][0]) & \
                            (w > cond['wRng'][1]) & \
                            (h < cond['hRng'][0]) & \
                            (h > cond['hRng'][1])
                self.labels[i][ignore_idx, 0] = -1

    def img2label_paths(self, img_paths):
        """Generates label file paths from corresponding image file paths by replacing `/images/{}` with `/labels/` and
        extension with `.txt`.
        """
        sa, sb = f"{os.sep}images{os.sep}{{}}{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Caches dataset labels, verifies images, reads shapes, and tracks dataset integrity."""
        x = {}  # dict
        data = []  # use list to keep image orders
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."

        pbar = tqdm(
            zip(self.im_files, self.label_files, repeat(prefix)),
            desc=desc,
            total=len(self.im_files),
            bar_format=TQDM_BAR_FORMAT)

        for blob in pbar:
            im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg = verify_rgbt_image_label(self.modalities, blob)
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if msg:
                msgs.append(msg)
            pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            data.append([im_file, lb, shape, segments])
        pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["data"] = data
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def prepare_image_cache(self, cache_images, prefix, n):
        # Cache images into RAM/disk for faster training
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [(None, None)] * n
        self.npy_files = [[Path(f.format(m)).with_suffix(".npy") for m in self.modalities] for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [(None, None)] * n, [(None, None)] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            results = ThreadPool(NUM_THREADS).imap(lambda i: (i, fcn(i)), self.indices)
            pbar = tqdm(results, total=len(self.indices), bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == "disk":
                    b += sum([f.stat().st_size for f in self.npy_files[i]])
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += sum([f.nbytes for f in self.ims[i]]) * WORLD_SIZE
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image to disk as an *.npy file for quicker loading, identified by index `i`."""
        for f, m in zip(self.npy_files[i], self.modalities):
            if not f.exists():
                np.save(f.as_posix(), cv2.imread(self.im_files[i].format(m)))

    def load_mosaic(self, index):
        # Loads 4 RGBT image mosaic into a single image, labels, and segments
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)

        # Initialize img4 for both modalities
        # Assuming imgs will return a tuple (lwir_img, visible_img)
        temp_imgs_modalities, _, _ = self.load_image(indices[0])
        temp_img_lwir_shape = temp_imgs_modalities[0].shape
        temp_img_vis_shape = temp_imgs_modalities[1].shape
        
        img4_lwir = np.full((s * 2, s * 2, temp_img_lwir_shape[2] if len(temp_img_lwir_shape) == 3 else 1), 114, dtype=np.uint8)
        img4_vis = np.full((s * 2, s * 2, temp_img_vis_shape[2]), 114, dtype=np.uint8)

        for i, index_loop_var in enumerate(indices): # 루프 변수명 변경
            # Load images for both modalities
            imgs_modalities, _, resized_shapes_per_modality = self.load_image(index_loop_var)
            img_lwir, img_vis = imgs_modalities[0], imgs_modalities[1]
            h, w = resized_shapes_per_modality[0]
            # 이제 resized_shapes_per_modality 변수가 정의되었으므로 아래 라인에서 사용 가능
            h, w = resized_shapes_per_modality[0]

            # place img in img4
            if i == 0:  # top left
                # img4_lwir, img4_vis 초기화는 루프 밖으로 이동했으므로 여기서는 좌표 계산만 수행
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4_lwir[y1a:y2a, x1a:x2a] = img_lwir[y1b:y2b, x1b:x2b]
            img4_vis[y1a:y2a, x1a:x2a] = img_vis[y1b:y2b, x1b:x2b]

            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                # normalized xywh to pixel xyxy format
                labels[:, 1:3] += labels[:, 3:5] / 2.0 # (x_lefttop, y_lefttop) -> (x_center, y_center) for proper scaling
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment - apply random_perspective to both modalities
        img4_lwir, labels4_transformed, M_transform = random_perspective(
            img4_lwir,
            labels4.copy(), # 원본 labels4를 변경하지 않도록 .copy() 사용 권장
            segments4.copy() if segments4 else [], # segments4도 복사본 사용 또는 빈 리스트 전달
            degrees=self.hyp.get("degrees", 0.0), # hyp 딕셔너리에 키가 없을 경우를 대비해 .get() 사용
            translate=self.hyp.get("translate", 0.0),
            scale=self.hyp.get("scale", 0.0),
            shear=self.hyp.get("shear", 0.0),
            perspective=self.hyp.get("perspective", 0.0),
            border=self.mosaic_border,
        )
        img4_vis, _, _ = random_perspective( 
            img4_vis,
            (), 
            (), 
            degrees=self.hyp.get("degrees", 0.0),
            translate=self.hyp.get("translate", 0.0),
            scale=self.hyp.get("scale", 0.0),
            shear=self.hyp.get("shear", 0.0),
            perspective=self.hyp.get("perspective", 0.0),
            border=self.mosaic_border,
            M=M_transform
        )
        
        p_copy_paste = self.hyp.get("copy_paste", 0.0)
        
        # copy_paste를 위한 최종 이미지와 레이블 변수 초기화
        final_img_lwir = img4_lwir
        final_img_vis = img4_vis
        final_labels = labels4_transformed # random_perspective 후의 레이블
        final_segments = segments4 # random_perspective 후의 세그먼트 (만약 수정되었다면)

        if p_copy_paste > 0:
            LOGGER.warning("RGBT Copy-Paste is active (p > 0) but using a non-RGBT aware function. Augmentations might be inconsistent.")
            

            final_img_lwir, final_labels, final_segments = copy_paste(
                img4_lwir, 
                labels4_transformed.copy(), # copy_paste에 전달할 때는 복사본 사용
                segments4.copy() if segments4 else [], 
                p=p_copy_paste
            )

            final_img_vis, _, _ = copy_paste(
                img4_vis, 
                labels4_transformed.copy(), # 변환 전 레이블을 사용할지, 첫번째 copy_paste 후 레이블을 사용할지 정책 필요
                                           # 여기서는 random_perspective 후의 레이블을 사용
                segments4.copy() if segments4 else [], 
                p=p_copy_paste
            )
            # 참고: 위와 같이 두 번 호출하면 각 이미지에 대해 다른 랜덤 샘플링이 copy_paste 내부에서 발생할 수 있습니다.

        return (final_img_lwir, final_img_vis), final_labels
    
# D:\AUE8088\utils\dataloaders.py LoadRGBTImagesAndLabels 클래스 내 __getitem__ 메소드

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        
        shapes = None  # shapes는 non-mosaic 경로에서 설정되므로 초기화
        imgs_to_return = None # 최종 반환될 이미지 텐서 리스트

        if mosaic:
            # Load mosaic for RGBT
            # self.load_mosaic는 ( (lwir_np, vis_np), labels_np_Nx6 )를 반환합니다.
            imgs_mosaic_np, labels_mosaic_np = self.load_mosaic(index) 
            
            # MixUp augmentation for RGBT
            if random.random() < hyp["mixup"]:
                # mixup 함수는 ((lwir1, vis1), labels1, (lwir2, vis2), labels2)를 받아
                # ((mixed_lwir, mixed_vis), combined_labels)를 반환합니다.
                imgs2_mosaic_np_tuple, labels2_mosaic_np = self.load_mosaic(random.choice(self.indices))
                imgs_mosaic_np, labels_mosaic_np = mixup(
                    imgs_mosaic_np, labels_mosaic_np, imgs2_mosaic_np_tuple, labels2_mosaic_np
                )

            nl = len(labels_mosaic_np)
            # labels_mosaic_np는 [cls, x, y, w, h, occ_level] 형태의 NumPy 배열입니다.
            # labels_out은 [batch_idx_placeholder, cls, x, y, w, h, occ_level] 형태의 Tensor가 됩니다.
            labels_out = torch.zeros((nl, 7)) 
            if nl:
                # 1) 클래스 번호
                labels_out[:, 1] = torch.from_numpy(labels_mosaic_np[:, 0])

                # 2) pixel xyxy → normalised xywh
                h_img, w_img = imgs_mosaic_np[0].shape[:2]   # 두 모달리티 크기는 동일
                labels_out[:, 2:6] = torch.from_numpy(
                    xyxy2xywhn(labels_mosaic_np[:, 1:5].copy(),
                            w=w_img, h=h_img, clip=True, eps=1e-3)
                )

                # 3) occlusion 값 있으면 보존
                if labels_mosaic_np.shape[1] == 6:
                    labels_out[:, 6] = torch.from_numpy(labels_mosaic_np[:, 5])
                        
            # 모자이크 NumPy 이미지들을 Tensor로 변환
            processed_mosaic_tensors = []
            for img_modality_np in imgs_mosaic_np: # imgs_mosaic_np는 (lwir_np, vis_np) 튜플
                img_tensor = img_modality_np.transpose((2, 0, 1))  # HWC to CHW
                if img_tensor.shape[0] == 3:  # 3채널 이미지의 경우 (BGR -> RGB)
                    img_tensor = img_tensor[::-1, :, :] 
                img_tensor = np.ascontiguousarray(img_tensor)
                processed_mosaic_tensors.append(torch.from_numpy(img_tensor))
                
            imgs_to_return = processed_mosaic_tensors # Tensor 리스트

        else: # Not mosaic
            # Load image
            # imgs_modalities_np: [(lwir_np, vis_np)], hw0s_list_of_tuples: [ (h0_lwir, w0_lwir), ...], ...
            imgs_modalities_np, hw0s_list, hw_resized_list = self.load_image(index) 
            
            processed_imgs_tensors = [] # 각 모달리티의 최종 처리된 텐서를 저장할 리스트
            # non-mosaic 경로에서 최종 레이블 (증강 적용 후, NumPy 형태)
            labels_processed_final_np = self.labels[index].copy() 
                                                            
            M_random_perspective = None # random_perspective 변환 행렬 공유용

            for ii, (img_np, hw0_tuple, hw_resized_tuple) in enumerate(zip(imgs_modalities_np, hw0s_list, hw_resized_list)):
                h0, w0 = hw0_tuple       # 원본 높이, 너비
                # h, w = hw_resized_tuple # 리사이즈된 높이, 너비 (letterbox 이전) -> 이 변수는 현재 사용되지 않음

                # Letterbox
                target_shape_for_letterbox = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
                img_lb_np, ratio_tuple, pad_tuple = letterbox(img_np.copy(), target_shape_for_letterbox, auto=False, scaleup=self.augment)
                
                if ii == 0: # COCO mAP 리스케일링을 위한 shapes 정보는 첫 번째 모달리티 기준으로 한 번만 설정
                    shapes = (h0, w0), (ratio_tuple, pad_tuple)

                # 현재 반복에서의 레이블 (매번 원본에서 복사하여 사용)
                current_labels_for_modality_np = self.labels[index].copy() # [cls, xc, yc, w, h, occ]
                
                # 레이블 좌표 변환: normalized xywh -> pixel xyxy (letterboxed 이미지 기준)
                # random_perspective 함수는 pixel xyxy 입력을 가정합니다.
                if current_labels_for_modality_np.size > 0:
                    # normalized [x_topleft, y_topleft, w, h] -> pixel [x1, y1, x2, y2]
                    # 이전의 잘못된 변환 코드를 아래 코드로 교체해야 합니다.
                    labels_on_orig_xyxy = np.zeros_like(current_labels_for_modality_np[:, :5]) # [cls, x1, y1, x2, y2]
                    labels_on_orig_xyxy[:, 0] = current_labels_for_modality_np[:, 0] # cls
                    labels_on_orig_xyxy[:, 1] = current_labels_for_modality_np[:, 1] * w0  # x1 = x_tl * w0
                    labels_on_orig_xyxy[:, 2] = current_labels_for_modality_np[:, 2] * h0  # y1 = y_tl * h0
                    labels_on_orig_xyxy[:, 3] = (current_labels_for_modality_np[:, 1] + current_labels_for_modality_np[:, 3]) * w0  # x2 = (x_tl + w) * w0
                    labels_on_orig_xyxy[:, 4] = (current_labels_for_modality_np[:, 2] + current_labels_for_modality_np[:, 4]) * h0  # y2 = (y_tl + h) * h0

                    # letterbox 스케일링 및 패딩 적용
                    labels_on_lb_xyxy = labels_on_orig_xyxy.copy()
                    labels_on_lb_xyxy[:, [1,3]] = labels_on_orig_xyxy[:, [1,3]] * ratio_tuple[0] + pad_tuple[0] # x coords
                    labels_on_lb_xyxy[:, [2,4]] = labels_on_orig_xyxy[:, [2,4]] * ratio_tuple[1] + pad_tuple[1] # y coords

                    current_labels_pixel_xyxy = labels_on_lb_xyxy
                else:
                    current_labels_pixel_xyxy = np.array([])

                # 증강 적용
                if self.augment:
                    img_aug_np = img_lb_np.copy() # 증강을 위한 이미지 복사
                    # labels_aug_xyxy는 증강 후의 [cls, x1,y1,x2,y2] 형태가 됩니다.
                    labels_aug_xyxy = current_labels_pixel_xyxy.copy() if current_labels_pixel_xyxy.size > 0 else np.array([])
                    
                    # segments 처리 (필요시, 여기서는 단순화를 위해 기본값 전달)
                    current_segments = self.segments[index].copy() if self.segments and self.segments[index] else []

                    if ii == 0: # 첫 번째 모달리티: random_perspective 변환 행렬 M 생성 및 적용
                        img_aug_np, labels_aug_xyxy, M_random_perspective = random_perspective(
                            img_aug_np,
                            labels_aug_xyxy,
                            current_segments,
                            degrees=hyp.get("degrees", 0.0),
                            translate=hyp.get("translate", 0.0),
                            scale=hyp.get("scale", 0.0),
                            shear=hyp.get("shear", 0.0),
                            perspective=hyp.get("perspective", 0.0)
                        )
                    else: # 두 번째 모달리티: 동일한 M 행렬 적용
                        img_aug_np, _, _ = random_perspective(img_aug_np, M=M_random_perspective)
                        # 레이블은 첫 번째 모달리티 기준으로 이미 변환됨

                    # Albumentations (random_perspective 후, pixel xyxy 레이블 사용)
                    if self.albumentations and labels_aug_xyxy.size > 0:
                        # Albumentations는 YOLO 포맷(normalized xc,yc,w,h)을 기대하므로 변환 필요
                        h_img_for_alb, w_img_for_alb = img_aug_np.shape[:2]
                        labels_for_alb_yolo = np.zeros((len(labels_aug_xyxy), 5))
                        if len(labels_aug_xyxy) > 0:
                            labels_for_alb_yolo[:,0] = labels_aug_xyxy[:,0] # class
                            labels_for_alb_yolo[:,1:] = xyxy2xywhn(labels_aug_xyxy[:,1:5].copy(), w=w_img_for_alb, h=h_img_for_alb, clip=True, eps=1e-3)

                        transformed = self.albumentations(
                            image=img_aug_np, # cv2.cvtColor(img_aug_np, cv2.COLOR_BGR2RGB) -> Albumentations 클래스 내부에서 처리 가정
                            bboxes=labels_for_alb_yolo[:, 1:].tolist(), # YOLO format bboxes
                            class_labels=labels_for_alb_yolo[:, 0].tolist()
                        )
                        img_aug_np = transformed['image']
                        
                        if ii == 0: # 레이블은 첫 번째 모달리티 기준으로만 업데이트
                            labels_from_alb_yolo = np.array(transformed['bboxes'])
                            if labels_from_alb_yolo.size > 0:
                                # Albumentations가 YOLO 포맷으로 반환한 것을 다시 pixel xyxy로 변환하여 labels_aug_xyxy 업데이트
                                labels_aug_xyxy[:,1:5] = xywhn2xyxy(labels_from_alb_yolo, w=img_aug_np.shape[1], h=img_aug_np.shape[0])
                            else:
                                labels_aug_xyxy = np.array([]) # 모든 박스가 사라진 경우


                    # HSV (Visible 이미지, 즉 ii == 1 일 때만 적용 가정)
                    if ii == 1: 
                        augment_hsv(img_aug_np, hgain=hyp.get("hsv_h",0.0), sgain=hyp.get("hsv_s",0.0), vgain=hyp.get("hsv_v",0.0))

                    # Flip (이미지에 적용 후, 레이블은 ii == 0 일 때만 업데이트)
                    # labels_aug_xyxy는 pixel xyxy 상태 [cls, x1,y1,x2,y2]
                    if random.random() < hyp.get("flipud",0.0):
                        img_aug_np = np.flipud(img_aug_np)
                        if ii == 0 and labels_aug_xyxy.size > 0:
                            labels_aug_xyxy[:, [2, 4]] = img_aug_np.shape[0] - labels_aug_xyxy[:, [4, 2]] # y1, y2 좌표 반전

                    if random.random() < hyp.get("fliplr",0.0):
                        img_aug_np = np.fliplr(img_aug_np)
                        if ii == 0 and labels_aug_xyxy.size > 0:
                            labels_aug_xyxy[:, [1, 3]] = img_aug_np.shape[1] - labels_aug_xyxy[:, [3, 1]] # x1, x2 좌표 반전
                    
                    if ii == 0: # 최종 증강된 레이블(pixel xyxy)을 labels_processed_final_np에 저장
                        labels_processed_final_np = labels_aug_xyxy

                    final_img_for_modality_np = img_aug_np
                else: # self.augment == False
                    final_img_for_modality_np = img_lb_np
                    if ii == 0: # 증강 안 할 시 원본 레이블(pixel xyxy) 사용
                        labels_processed_final_np = current_labels_pixel_xyxy


                # 이미지 NumPy 배열을 Tensor로 변환
                img_tensor = final_img_for_modality_np.transpose((2, 0, 1))  # HWC to CHW
                if img_tensor.shape[0] == 3: # 3채널 이미지 (BGR -> RGB)
                    img_tensor = img_tensor[::-1, :, :]
                img_tensor = np.ascontiguousarray(img_tensor)
                processed_imgs_tensors.append(torch.from_numpy(img_tensor))
            
            # non-mosaic 경로의 최종 레이블 처리 (pixel xyxy -> normalized xywh 및 occlusion 추가)
            nl = len(labels_processed_final_np)
            labels_out = torch.zeros((nl, 7))  # batch_idx_placeholder, cls, xc, yc, w, h, occ_level
            if nl:
                # cls 저장
                labels_out[:, 1] = torch.from_numpy(labels_processed_final_np[:, 0])
                # pixel xyxy -> normalized xywh 변환
                # 마지막으로 사용된 final_img_for_modality_np의 shape 기준으로 정규화 (모든 모달리티가 동일한 최종 shape 가정)
                h_final_img, w_final_img = final_img_for_modality_np.shape[:2]
                labels_out[:, 2:6] = torch.from_numpy(xyxy2xywhn(labels_processed_final_np[:, 1:5].copy(), w=w_final_img, h=h_final_img, clip=True, eps=1e-3))
                
                # Occlusion level 추가 (원본 레이블에서 가져옴)
                if self.labels[index].shape[1] == 6 and len(self.labels[index]) == nl: # 원본에 occlusion 정보가 있고 길이가 같다면
                    labels_out[:, 6] = torch.from_numpy(self.labels[index][:, 5])
                else: # 없다면 0으로 채움 (또는 다른 기본값)
                    labels_out[:, 6] = 0

            imgs_to_return = processed_imgs_tensors


        # 공통 로직: labels_out의 마지막 컬럼 (occlusion level) 제거
        # labels_out은 (N, 7) [batch_idx_placeholder, cls, xc, yc, w, h, occ] 형태를 가집니다.
        if labels_out.shape[1] == 7:
             labels_out = labels_out[:, :-1] # 마지막 occlusion level 컬럼 제거 -> (N, 6)
        elif nl > 0 and labels_out.shape[1] != 6: # nl > 0 인데 컬럼 수가 6이 아니면 경고 (이미 6개면 그대로 둠)
            LOGGER.warning(f"labels_out has unexpected shape {labels_out.shape} before final processing for index {self.im_files[index]}. Expected 6 or 7 columns.")
        # 만약 nl == 0 이면 labels_out은 (0,7) 또는 (0,6) 이고, 슬라이싱은 문제 없음.

        return imgs_to_return, labels_out, self.im_files[index], shapes, index

    def load_image(self, i):
        """
        Loads an image by index, returning the image, its original dimensions, and resized dimensions.

        Returns (im, original hw, resized hw)
        """
        imgs, f, fns = (
            self.ims[i],
            self.im_files[i],
            self.npy_files[i],
        )

        if any(img is None for img in imgs):  # not cached in RAM
            if all(fn.exists() for fn in fns):  # load npy
                imgs = [np.load(fn) for fn in fns]
            else:  # read image
                imgs = [cv2.imread(f.format(m)) for m in self.modalities]  # BGR
                assert all(img is not None for img in imgs), f"Image Not Found {f}"

            h0s, w0s = [], []
            img_shapes = []
            for i, img in enumerate(imgs):
                h0, w0 = img.shape[:2]  # orig hw
                r = self.img_size / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                    imgs[i] = cv2.resize(img, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
                h0s.append(h0)
                w0s.append(w0)
                img_shapes.append(imgs[i].shape[:2])
            return imgs, (h0s, w0s), img_shapes

        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized


    @staticmethod
    def collate_fn(batch):
        """Batches images, labels, paths, shapes, and indices assigning unique indices to targets in merged label tensor."""
        imgs, label, path, shapes, indices = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        imgs_lwir = torch.stack([img[0] for img in imgs], 0)
        imgs_vis = torch.stack([img[1] for img in imgs], 0)
        return (imgs_lwir, imgs_vis), torch.cat(label, 0), path, shapes, indices

    @staticmethod
    def collate_fn4(batch):
        raise NotImplementedError

# Ancillary functions --------------------------------------------------------------------------------------------------
def flatten_recursive(path=DATASETS_DIR / "coco128"):
    """Flattens a directory by copying all files from subdirectories to a new top-level directory, preserving
    filenames.
    """
    new_path = Path(f"{str(path)}_flat")
    if os.path.exists(new_path):
        shutil.rmtree(new_path)  # delete output folder
    os.makedirs(new_path)  # make new output folder
    for file in tqdm(glob.glob(f"{str(Path(path))}/**/*.*", recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / "coco128"):
    """
    Converts a detection dataset to a classification dataset, creating a directory for each class and extracting
    bounding boxes.

    Example: from utils.dataloaders import *; extract_boxes()
    """
    path = Path(path)  # images dir
    shutil.rmtree(path / "classification") if (path / "classification").is_dir() else None  # remove existing
    files = list(path.rglob("*.*"))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / "classification") / f"{c}" / f"{path.stem}_{im_file.stem}_{j}.jpg"  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1] : b[3], b[0] : b[2]]), f"box failure in {f}"


def autosplit(path=DATASETS_DIR / "coco128/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file


def verify_image_label(args):
    """Verifies a single image-label pair, ensuring image format, size, and legal label values."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


def verify_rgbt_image_label(modalities, args):
    """Verifies a single image-label pair, ensuring image format, size, and legal label values."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
    try:
        for modality in modalities:
            # verify images
            im = Image.open(im_file.format(modality))
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                        msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                # KAIST dataset labels: N x 6 (cls, x_lefttop, y_lefttop, width, height, occlevel)
                assert lb.shape[1] == 6, f"labels require 6 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:-1] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:-1][lb[:, 1:-1] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 6), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 6), dtype=np.float32)

        # assuming lwir and vis images have same shape!
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg

    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file} : ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]

class HUBDatasetStats:
    """
    Class for generating HUB dataset JSON and `-hub` dataset directory.

    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally

    Usage
        from utils.dataloaders import HUBDatasetStats
        stats = HUBDatasetStats('coco128.yaml', autodownload=True)  # usage 1
        stats = HUBDatasetStats('path/to/coco128.zip')  # usage 2
        stats.get_json(save=False)
        stats.process_images()
    """

    def __init__(self, path="coco128.yaml", autodownload=False):
        """Initializes HUBDatasetStats with optional auto-download for datasets, given a path to dataset YAML or ZIP
        file.
        """
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            with open(check_yaml(yaml_path), errors="ignore") as f:
                data = yaml.safe_load(f)  # data dict
                if zipped:
                    data["path"] = data_dir
        except Exception as e:
            raise Exception("error/HUB/dataset_stats/yaml_load") from e

        check_dataset(data, autodownload)  # download dataset if missing
        self.hub_dir = Path(data["path"] + "-hub")
        self.im_dir = self.hub_dir / "images"
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {"nc": data["nc"], "names": list(data["names"].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _find_yaml(dir):
        """Finds and returns the path to a single '.yaml' file in the specified directory, preferring files that match
        the directory name.
        """
        files = list(dir.glob("*.yaml")) or list(dir.rglob("*.yaml"))  # try root level first and then recursive
        assert files, f"No *.yaml file found in {dir}"
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f"Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed"
        assert len(files) == 1, f"Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}"
        return files[0]

    def _unzip(self, path):
        """Unzips a .zip file at 'path', returning success status, unzipped directory, and path to YAML file within."""
        if not str(path).endswith(".zip"):  # path is data.yaml
            return False, None, path
        assert Path(path).is_file(), f"Error unzipping {path}, file not found"
        unzip_file(path, path=path.parent)
        dir = path.with_suffix("")  # dataset directory == zip name
        assert dir.is_dir(), f"Error unzipping {path}, {dir} not found. path/to/abc.zip MUST unzip to path/to/abc/"
        return True, str(dir), self._find_yaml(dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f, max_dim=1920):
        """Resizes and saves an image at reduced quality for web/app viewing, supporting both PIL and OpenCV."""
        f_new = self.im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, "JPEG", quality=50, optimize=True)  # save
        except Exception as e:  # use OpenCV
            LOGGER.info(f"WARNING ⚠️ HUB ops PIL failure {f}: {e}")
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    def get_json(self, save=False, verbose=False):
        """Generates dataset JSON for Ultralytics HUB, optionally saves or prints it; save=bool, verbose=bool."""

        def _round(labels):
            """Rounds class labels to integers and coordinates to 4 decimal places for improved label accuracy."""
            return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

        for split in "train", "val", "test":
            if self.data.get(split) is None:
                self.stats[split] = None  # i.e. no test set
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            x = np.array(
                [
                    np.bincount(label[:, 0].astype(int), minlength=self.data["nc"])
                    for label in tqdm(dataset.labels, total=dataset.n, desc="Statistics")
                ]
            )  # shape(128x80)
            self.stats[split] = {
                "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                "image_stats": {
                    "total": dataset.n,
                    "unlabelled": int(np.all(x == 0, 1).sum()),
                    "per_class": (x > 0).sum(0).tolist(),
                },
                "labels": [{str(Path(k).name): _round(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)],
            }

        # Save, print and return
        if save:
            stats_path = self.hub_dir / "stats.json"
            print(f"Saving {stats_path.resolve()}...")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            print(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        """Compresses images for Ultralytics HUB across 'train', 'val', 'test' splits and saves to specified
        directory.
        """
        for split in "train", "val", "test":
            if self.data.get(split) is None:
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            desc = f"{split} images"
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(self._hub_ops, dataset.im_files), total=dataset.n, desc=desc):
                pass
        print(f"Done. All images saved to {self.im_dir}")
        return self.im_dir


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.

    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        """Initializes YOLOv5 Classification Dataset with optional caching, augmentations, and transforms for image
        classification.
        """
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == "ram"
        self.cache_disk = cache == "disk"
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        """Fetches and transforms an image sample by index, supporting RAM/disk caching and Augmentations."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"]
        else:
            sample = self.torch_transforms(im)
        return sample, j


def create_classification_dataloader(
    path, imgsz=224, batch_size=16, augment=True, cache=False, rank=-1, workers=8, shuffle=True
):
    # Returns Dataloader object to be used with YOLOv5 Classifier
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=generator,
    )  # or DataLoader(persistent_workers=True)