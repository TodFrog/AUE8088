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
    random_perspective,
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
        """Fetches the dataset item at the given index, considering linear, shuffled, or weighted sampling."""
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and (random.random() < hyp["mosaic"] if hyp else False)
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *self.load_mosaic(random.choice(self.indices)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, index

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
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
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
                
# D:\AUE8088\utils\dataloaders.py LoadRGBTImagesAndLabels 클래스 내 load_mosaic 메서드

    def load_mosaic(self, index):
        # Loads 4 RGBT image mosaic into a single image, labels, and segments
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)

        temp_imgs_modalities, _, _ = self.load_image(indices[0])
        temp_img_lwir_shape = temp_imgs_modalities[0].shape
        temp_img_vis_shape = temp_imgs_modalities[1].shape
        
        img4_lwir = np.full((s * 2, s * 2, temp_img_lwir_shape[2] if len(temp_img_lwir_shape) == 3 else 1), 114, dtype=np.uint8)
        img4_vis = np.full((s * 2, s * 2, temp_img_vis_shape[2]), 114, dtype=np.uint8)

        degrees = self.hyp.get("degrees", 0.0) if self.hyp else 0.0
        translate = self.hyp.get("translate", 0.0) if self.hyp else 0.0
        scale = self.hyp.get("scale", 0.0) if self.hyp else 0.0
        shear = self.hyp.get("shear", 0.0) if self.hyp else 0.0
        perspective = self.hyp.get("perspective", 0.0) if self.hyp else 0.0
        copy_paste_p = self.hyp.get("copy_paste", 0.0) if self.hyp else 0.0


        for i, index_loop_var in enumerate(indices): # 루프 변수명 변경
            # Load images for both modalities
            imgs_modalities, _, resized_shapes_per_modality = self.load_image(index_loop_var)
            img_lwir, img_vis = imgs_modalities[0], imgs_modalities[1]
            h, w = resized_shapes_per_modality[0] # letterboxed 전 이미지 크기 (resize만 된 상태)

            # place img in img4
            if i == 0:  # top left
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

            # Labels (labels는 [cls, xc_norm, yc_norm, w_norm, h_norm, occlevel] 형태)
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                # labels는 현재 정규화된 xc,yc,w,h 형태입니다.
                # random_perspective는 픽셀 xyxy를 기대하므로 변환해야 합니다.
                # 먼저 정규화된 xc,yc,w,h를 정규화된 xyxy로 변환
                labels_norm_xyxy = xywhn2xyxy(labels[:, 1:5].copy(), w=1, h=1) # w=1, h=1로 정규화된 값을 유지
                
                # 정규화된 xyxy를 (resized_shapes_per_modality[0]의 w, h) 기준 픽셀 xyxy로 변환
                # 그리고 패딩을 더해서 mosaic 이미지 내에서의 픽셀 좌표로 변환
                labels_pixel_xyxy = np.zeros_like(labels_norm_xyxy)
                labels_pixel_xyxy[:, [0, 2]] = labels_norm_xyxy[:, [0, 2]] * w + padw
                labels_pixel_xyxy[:, [1, 3]] = labels_norm_xyxy[:, [1, 3]] * h + padh
                
                # segments는 정규화된 좌표이므로, 동일한 방식으로 픽셀 좌표로 변환 (패딩 고려)
                segments_pixel = [xyn2xy(x, w, h, padw, padh) for x in segments]

                # labels 변수 (labels4에 추가될 것)를 [cls, x1_pixel, y1_pixel, x2_pixel, y2_pixel, occlevel] 형태로 업데이트
                labels_to_append = np.zeros((labels.shape[0], 6))
                labels_to_append[:, 0] = labels[:, 0] # cls
                labels_to_append[:, 1:5] = labels_pixel_xyxy # x1, y1, x2, y2 (pixel)
                labels_to_append[:, 5] = labels[:, 5] # occlevel
                
                labels4.append(labels_to_append)
                segments4.extend(segments_pixel) # pixel segments 추가
            else:
                labels4.append(np.zeros((0,6), dtype=np.float32)) # 빈 레이블도 6개 컬럼 유지 (cls, x1,y1,x2,y2,occlevel)


        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0) # labels4는 이제 [cls, x1_pixel, y1_pixel, x2_pixel, y2_pixel, occlevel]
        # random_perspective에 전달될 box 부분만 클리핑
        for x in (labels4[:, 1:5], *segments4): # segments4는 이미 pixel 좌표
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment - apply random_perspective to both modalities
        # random_perspective는 [cls, x1,y1,x2,y2] (픽셀) 레이블을 기대
        img4_lwir, labels4_transformed, M_transform = random_perspective(
            img4_lwir,
            labels4.copy()[:, :5], # random_perspective는 [cls, x1,y1,x2,y2]만 기대, occlevel 제거
            segments4.copy() if segments4 else [], 
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            border=self.mosaic_border,
        )
        img4_vis, _, _ = random_perspective( 
            img4_vis,
            (), # 레이블은 첫 번째 모달리티에서만 변환
            (), # 세그먼트도 마찬가지
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            border=self.mosaic_border,
            M=M_transform
        )
        
        # random_perspective에서 반환된 labels4_transformed는 [cls, x1,y1,x2,y2] (픽셀) 입니다.
        # 여기에 occlusion level을 다시 붙여야 합니다.
        # 이 과정에서 레이블의 개수가 변경될 수 있으므로, 원본 labels4의 occlevel을 그대로 가져오는 것은 위험합니다.
        # 간단하게 0으로 채우거나, 복잡한 매칭 로직이 필요합니다. 일단 0으로 채우는 것을 제안합니다.
        # Miss Rate 계산을 위해 occlevel이 필요한 경우, kaisteval.py에서 처리해야 할 수도 있습니다.
        
        # labels4_transformed는 [cls, x1, y1, x2, y2] (픽셀)
        # labels4 (원본) 에는 6번째 컬럼에 occlevel이 있었음.
        # random_perspective에서 레이블이 필터링될 수 있으므로, occlevel을 다시 가져오기가 어렵습니다.
        # labels4_transformed 에 0으로 된 occlevel 컬럼을 추가합니다.
        if labels4_transformed.size > 0:
            labels4_transformed_with_occ = np.concatenate((labels4_transformed, np.zeros((len(labels4_transformed), 1))), axis=1)
        else:
            labels4_transformed_with_occ = np.zeros((0,6), dtype=np.float32) # 빈 배열도 6개 컬럼 유지
        
        # copy_paste를 위한 최종 이미지와 레이블 변수 초기화
        final_img_lwir = img4_lwir
        final_img_vis = img4_vis
        final_labels = labels4_transformed_with_occ # random_perspective 후의 레이블 (occlevel 포함)
        final_segments = segments4 # random_perspective 후의 세그먼트 (만약 수정되었다면)

        if copy_paste_p > 0: 
            LOGGER.warning("RGBT Copy-Paste is active (p > 0) but using a non-RGBT aware function. Augmentations might be inconsistent.")
            
            # copy_paste는 [im, labels, segments]를 받습니다. labels는 [cls, x1,y1,x2,y2] 형식.
            # labels4_transformed_with_occ는 [cls, x1,y1,x2,y2,occlevel]이므로, copy_paste에 전달할 때는 occlevel을 제외합니다.
            copied_img_lwir, copied_labels, copied_segments = copy_paste(
                img4_lwir, 
                final_labels[:, :5].copy(), # copy_paste는 [cls, x1,y1,x2,y2]만 받습니다.
                final_segments.copy() if final_segments else [], 
                p=copy_paste_p
            )
            # copy_paste 후에도 occlevel을 다시 붙여야 합니다. 여기서는 단순히 0으로 채웁니다.
            if copied_labels.size > 0:
                final_labels = np.concatenate((copied_labels, np.zeros((len(copied_labels), 1))), axis=1)
            else:
                final_labels = np.zeros((0,6), dtype=np.float32)
            final_img_lwir = copied_img_lwir
            final_segments = copied_segments # segments도 업데이트

            # visible 이미지에 대해서도 copy_paste (레이블은 첫 번째 모달리티가 주도)
            # copy_paste는 이미지에만 영향을 미치도록 호출 (레이블/세그먼트는 빈 튜플)
            final_img_vis, _, _ = copy_paste(
                img4_vis, 
                (), # 레이블 비워둠
                (), # 세그먼트 비워둠
                p=copy_paste_p
            )

        return (final_img_lwir, final_img_vis), final_labels # final_labels는 [cls, x1, y1, x2, y2, occlevel]
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp
        
        # hyp가 None일 경우, mosaic는 항상 False가 되도록 처리
        mosaic = self.mosaic and (random.random() < (hyp.get("mosaic", 0.0) if hyp else 0.0))

        shapes = None  # shapes는 non-mosaic 경로에서 설정되므로 초기화
        imgs_to_return = None # 최종 반환될 이미지 텐서 리스트

        if mosaic:
            # Load mosaic for RGBT
            imgs_mosaic_np, labels_mosaic_np = self.load_mosaic(index) 
            
            # MixUp augmentation for RGBT
            if random.random() < (hyp.get("mixup", 0.0) if hyp else 0.0):
                imgs2_mosaic_np_tuple, labels2_mosaic_np = self.load_mosaic(random.choice(self.indices))
                imgs_mosaic_np, labels_mosaic_np = mixup(
                    imgs_mosaic_np, labels_mosaic_np, imgs2_mosaic_np_tuple, labels2_mosaic_np
                )

            nl = len(labels_mosaic_np)
            # labels_mosaic_np는 [cls, x1, y1, x2, y2, occ_level] (픽셀) 형태일 것입니다 (load_mosaic 반환 형식).
            # labels_out은 [batch_idx_placeholder, cls, xc, yc, w, h] 형태의 Tensor가 됩니다.
            labels_out = torch.zeros((nl, 7)) # 최종적으로 occlevel도 담을 수 있게 7개 컬럼 유지
            if nl:
                # 1) 클래스 번호
                labels_out[:, 1] = torch.from_numpy(labels_mosaic_np[:, 0])

                # 2) pixel xyxy → normalised xywh
                h_img, w_img = imgs_mosaic_np[0].shape[:2]   # 두 모달리티 크기는 동일
                labels_out[:, 2:6] = torch.from_numpy(
                    xyxy2xywhn(labels_mosaic_np[:, 1:5].copy(), # labels_mosaic_np의 xyxy (픽셀) 부분
                            w=w_img, h=h_img)
                )

                # 3) occlusion 값 있으면 보존 (load_mosaic에서 반환된 labels_mosaic_np에 occlevel이 있다면)
                if labels_mosaic_np.shape[1] == 6: # load_mosaic에서 occlevel을 유지했다면
                    labels_out[:, 6] = torch.from_numpy(labels_mosaic_np[:, 5]).float()
                else:
                    labels_out[:, 6] = torch.zeros(nl).float()
                        
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
            imgs_modalities_np, hw0s_list, hw_resized_list = self.load_image(index) 
            
            processed_imgs_tensors = [] # 각 모달리티의 최종 처리된 텐서를 저장할 리스트
            # non-mosaic 경로에서 최종 레이블 (증강 적용 후, NumPy 형태)
            # labels_processed_final_np는 (N,6) [cls, xc_norm, yc_norm, w_norm, h_norm, occ_level] 입니다.
            # 이 상태에서 시작하여 각 증강 단계에서 올바른 형식으로 변환되어야 합니다.
            labels_original_normalized = self.labels[index].copy() # [cls, xc_norm, yc_norm, w_norm, h_norm, occ_level]

            M_random_perspective = None # random_perspective 변환 행렬 공유용

            degrees = hyp.get("degrees", 0.0) if hyp else 0.0
            translate = hyp.get("translate", 0.0) if hyp else 0.0
            scale = hyp.get("scale", 0.0) if hyp else 0.0
            shear = hyp.get("shear", 0.0) if hyp else 0.0
            perspective = hyp.get("perspective", 0.0) if hyp else 0.0
            hsv_h = hyp.get("hsv_h", 0.0) if hyp else 0.0
            hsv_s = hyp.get("hsv_s", 0.0) if hyp else 0.0
            hsv_v = hyp.get("hsv_v", 0.0) if hyp else 0.0
            flipud_p = hyp.get("flipud", 0.0) if hyp else 0.0 # _p 접미사 추가하여 중복 방지
            fliplr_p = hyp.get("fliplr", 0.0) if hyp else 0.0 # _p 접미사 추가하여 중복 방지

            flipud_rand = random.random() < flipud_p 
            fliplr_rand = random.random() < fliplr_p 

            labels_processed_final_np = labels_original_normalized.copy() # 이 변수는 증강 후 최종 레이블을 저장합니다.

            for ii, (img_np, hw0_tuple, hw_resized_tuple) in enumerate(zip(imgs_modalities_np, hw0s_list, hw_resized_list)):
                h0, w0 = hw0_tuple       # 원본 높이, 너비

                # Letterbox
                target_shape_for_letterbox = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
                img_lb_np, ratio_tuple, pad_tuple = letterbox(img_np.copy(), target_shape_for_letterbox, auto=False, scaleup=self.augment)
                
                if ii == 0: # COCO mAP 리스케일링을 위한 shapes 정보는 첫 번째 모달리티 기준으로 한 번만 설정
                    shapes = (h0, w0), (ratio_tuple, pad_tuple)

                # 현재 반복에서의 레이블 (ii == 0일 때만 실제 레이블 변환을 수행하고, ii == 1일 때는 동일한 변환 행렬 사용)
                # labels_original_normalized는 [cls, xc_norm, yc_norm, w_norm, h_norm, occ_level]
                
                # random_perspective는 pixel xyxy 입력을 기대하므로 변환합니다.
                current_labels_for_rp = np.array([])
                if labels_original_normalized.size > 0:
                    # [cls, xc_norm, yc_norm, w_norm, h_norm] 에서 [cls, x1_pixel, y1_pixel, x2_pixel, y2_pixel]로 변환
                    # letterbox 적용된 이미지 크기 (img_lb_np.shape[1], img_lb_np.shape[0]) 기준으로 픽셀 변환
                    labels_norm_xyxy = xywhn2xyxy(labels_original_normalized[:, 1:5].copy(), w=1, h=1) # 정규화된 xywh -> 정규화된 xyxy
                    
                    labels_pixel_xyxy = np.zeros_like(labels_norm_xyxy)
                    labels_pixel_xyxy[:, [0, 2]] = labels_norm_xyxy[:, [0, 2]] * img_lb_np.shape[1] # x1, x2 (pixel)
                    labels_pixel_xyxy[:, [1, 3]] = labels_norm_xyxy[:, [1, 3]] * img_lb_np.shape[0] # y1, y2 (pixel)
                    
                    current_labels_for_rp = np.concatenate((labels_original_normalized[:, 0:1], labels_pixel_xyxy), axis=1) # [cls, x1_pixel, y1_pixel, x2_pixel, y2_pixel]
                
                # segments도 pixel 좌표로 변환 (self.segments[index]는 정규화된 좌표로 가정)
                current_segments_pixel = []
                if self.segments and self.segments[index]:
                    for seg in self.segments[index]:
                        seg_pixel = np.copy(seg)
                        seg_pixel[:, 0] = seg_pixel[:, 0] * img_lb_np.shape[1] # x (pixel)
                        seg_pixel[:, 1] = seg_pixel[:, 1] * img_lb_np.shape[0] # y (pixel)
                        current_segments_pixel.append(seg_pixel)


                # 증강 적용
                if self.augment:
                    img_aug_np_current = img_lb_np.copy() # 증강을 위한 이미지 복사
                    # labels_aug_xyxy_current는 random_perspective 이후의 [cls, x1,y1,x2,y2] (픽셀) 형태가 됩니다.
                    labels_aug_xyxy_current = current_labels_for_rp.copy() if current_labels_for_rp.size > 0 else np.array([])
                    
                    if ii == 0: # 첫 번째 모달리티: random_perspective 변환 행렬 M 생성 및 적용
                        img_aug_np_current, labels_aug_xyxy_current, M_random_perspective = random_perspective(
                            img_aug_np_current,
                            labels_aug_xyxy_current,
                            current_segments_pixel,
                            degrees=degrees,
                            translate=translate,
                            scale=scale,
                            shear=shear,
                            perspective=perspective
                        )
                    else: # 두 번째 모달리티: 동일한 M 행렬 적용
                        img_aug_np_current, _, _ = random_perspective(img_aug_np_current, M=M_random_perspective)
                        # 레이블은 ii == 0일 때 이미 labels_aug_xyxy_current에 반영되었으므로, 여기서는 레이블 업데이트 없음.


                    # Albumentations (random_perspective 후, pixel xyxy 레이블 사용)
                    # Albumentations는 BGR 이미지를 입력받아 RGB로 처리 후 RGB로 반환하고, 레이블은 YOLO 형식 (normalized xc,yc,w,h)으로 받음/반환함.
                    if self.albumentations and labels_aug_xyxy_current.size > 0:
                        # labels_aug_xyxy_current [cls, x1,y1,x2,y2] (픽셀) -> labels_for_alb_yolo [cls, xc_norm, yc_norm, w_norm, h_norm]
                        h_img_for_alb, w_img_for_alb = img_aug_np_current.shape[:2]
                        labels_for_alb_yolo = np.zeros((len(labels_aug_xyxy_current), 5))
                        labels_for_alb_yolo[:,0] = labels_aug_xyxy_current[:,0] # class
                        labels_for_alb_yolo[:,1:] = xyxy2xywhn(labels_aug_xyxy_current[:,1:5].copy(), w=w_img_for_alb, h=h_img_for_alb)

                        # Albumentations.__call__은 위치 인자 (im, labels)를 받음
                        img_alb_output, labels_alb_output = self.albumentations(
                            cv2.cvtColor(img_aug_np_current, cv2.COLOR_BGR2RGB), # BGR -> RGB
                            labels_for_alb_yolo # [cls, xc_norm, yc_norm, w_norm, h_norm]
                        )
                        img_aug_np_current = cv2.cvtColor(img_alb_output, cv2.COLOR_RGB2BGR) # RGB -> BGR

                        if ii == 0: # 레이블은 첫 번째 모달리티 기준으로만 업데이트
                            if labels_alb_output.size > 0:
                                # labels_alb_output은 [cls, xc_norm, yc_norm, w_norm, h_norm]
                                # 이를 다시 pixel xyxy로 변환하여 labels_aug_xyxy_current 업데이트
                                labels_aug_xyxy_current_updated = np.zeros((len(labels_alb_output), 5))
                                labels_aug_xyxy_current_updated[:,0] = labels_alb_output[:,0] # class
                                labels_aug_xyxy_current_updated[:,1:5] = xywhn2xyxy(labels_alb_output[:,1:5].copy(), w=img_aug_np_current.shape[1], h=img_aug_np_current.shape[0])
                                labels_aug_xyxy_current = labels_aug_xyxy_current_updated
                            else:
                                labels_aug_xyxy_current = np.array([]) # 모든 박스가 사라진 경우


                    # HSV (Visible 이미지, 즉 ii == 1 일 때만 적용)
                    if ii == 1: 
                        augment_hsv(img_aug_np_current, hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)


                    # Flip (모든 모달리티 이미지에 적용 후, 레이블은 ii == 0 일 때만 업데이트)
                    # labels_aug_xyxy_current는 pixel xyxy 상태 [cls, x1,y1,x2,y2]
                    if flipud_rand: # 미리 생성된 랜덤 값 사용
                        img_aug_np_current = np.flipud(img_aug_np_current)
                        if ii == 0 and labels_aug_xyxy_current.size > 0:
                            # y1, y2 좌표를 이미지 높이에 맞춰 반전
                            labels_aug_xyxy_current[:, [2, 4]] = img_aug_np_current.shape[0] - labels_aug_xyxy_current[:, [4, 2]] 

                    if fliplr_rand: # 미리 생성된 랜덤 값 사용
                        img_aug_np_current = np.fliplr(img_aug_np_current)
                        if ii == 0 and labels_aug_xyxy_current.size > 0:
                            # x1, x2 좌표를 이미지 너비에 맞춰 반전
                            labels_aug_xyxy_current[:, [1, 3]] = img_aug_np_current.shape[1] - labels_aug_xyxy_current[:, [3, 1]] 
                    
                    if ii == 0: # 최종 증강된 레이블(pixel xyxy)을 labels_processed_final_np에 저장
                        labels_processed_final_np = labels_aug_xyxy_current # pixel xyxy

                    final_img_for_modality_np = img_aug_np_current
                else: # self.augment == False (증강을 하지 않는 경우)
                    final_img_for_modality_np = img_lb_np
                    if ii == 0: # 증강 안 할 시 원본 레이블(letterbox 적용된 pixel xyxy) 사용
                        # current_labels_for_rp는 이미 letterbox 적용 후 pixel xyxy 형태
                        labels_processed_final_np = current_labels_for_rp.copy()


                # 이미지 NumPy 배열을 Tensor로 변환
                img_tensor = final_img_for_modality_np.transpose((2, 0, 1))  # HWC to CHW
                if img_tensor.shape[0] == 3: # 3채널 이미지 (BGR -> RGB)
                    img_tensor = img_tensor[::-1, :, :]
                img_tensor = np.ascontiguousarray(img_tensor)
                processed_imgs_tensors.append(torch.from_numpy(img_tensor))
            
            # non-mosaic 경로의 최종 레이블 처리 (pixel xyxy -> normalized xywh 및 occlusion 추가)
            nl = len(labels_processed_final_np) # labels_processed_final_np는 [cls, x1, y1, x2, y2] (픽셀)
            labels_out = torch.zeros((nl, 7))  # batch_idx_placeholder, cls, xc, yc, w, h, occ_level
            if nl:
                # cls 저장
                labels_out[:, 1] = torch.from_numpy(labels_processed_final_np[:, 0]).float() # 클래스 ID는 float으로 저장될 수 있음

                # pixel xyxy -> normalized xywh 변환
                h_final_img, w_final_img = final_img_for_modality_np.shape[:2]
                labels_out[:, 2:6] = torch.from_numpy(xyxy2xywhn(labels_processed_final_np[:, 1:5].copy(), w=w_final_img, h=h_final_img)).float()
                
                # Occlusion level 추가 (원본 레이블에서 가져옴)
                if self.labels[index].shape[1] == 6: # 원본 레이블에 occlusion 정보가 있다면 (6번째 컬럼이 존재)
                    if len(self.labels[index]) == nl:
                        # 원본 레이블의 occlusion 컬럼을 직접 사용
                        labels_out[:, 6] = torch.from_numpy(self.labels[index][:, 5]).float()
                    else: # 레이블 수가 변경되었다면 (예: 일부 레이블이 잘려나감), 0으로 채웁니다.
                        labels_out[:, 6] = torch.zeros(nl).float()
                else: # 원본 레이블에 occlusion 정보가 없다면 0으로 채움
                    labels_out[:, 6] = torch.zeros(nl).float()

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


def verify_rgbt_image_label(modalities, args):
    """Verifies a single image-label pair for RGBT datasets, ensuring image format, size, and legal label values."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []
    try:
        # 이미지의 실제 해상도 가져오기 (W, H) - 첫 번째 모달리티를 기준으로 함
        im_path_for_shape = im_file.format(modalities[0])
        im_pil = Image.open(im_path_for_shape)
        im_pil.verify()
        actual_img_w, actual_img_h = exif_size(im_pil) 
        
        # 모든 모달리티 이미지 확인 (형식, 크기, 손상 여부)
        for modality in modalities:
            im = Image.open(im_file.format(modality))
            im.verify()
            shape = exif_size(im) # (W, H)
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file.format(modality), "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":
                        ImageOps.exif_transpose(Image.open(im_file.format(modality))).save(im_file.format(modality), "JPEG", subsampling=0, quality=100)
                        msg = f"{prefix}WARNING ⚠️ {im_file.format(modality)}: corrupt JPEG restored and saved"

        # 레이블 파일 확인 및 로드
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32) # (cls, x_lt_norm, y_lt_norm, w_norm, h_norm, occlevel)
            nl = len(lb)
            if nl:
                # KAIST dataset labels: N x 6 (cls, x_lefttop_norm, y_lefttop_norm, width_norm, height_norm, occlevel)
                assert lb.shape[1] == 6, f"labels require 6 columns, {lb.shape[1]} columns detected for {lb_file}"

                # ===== START: KAIST 레이블 (x_lefttop_norm, y_lefttop_norm, width_norm, height_norm)을 YOLOv5 (xc_norm, yc_norm, w_norm, h_norm)로 변환 =====
                # 이 변환은 원본 KAIST 레이블이 이미 정규화된 비율이라고 가정합니다.
                # xc = x_lefttop + width / 2
                # yc = y_lefttop + height / 2
                
                xc_norm = lb[:, 1] + lb[:, 3] / 2
                yc_norm = lb[:, 2] + lb[:, 4] / 2
                w_norm = lb[:, 3]
                h_norm = lb[:, 4]

                # 새로운 레이블을 만듭니다: (cls, xc_norm, yc_norm, w_norm, h_norm, occlevel)
                new_lb = np.zeros_like(lb)
                new_lb[:, 0] = lb[:, 0] # class_id
                new_lb[:, 1] = xc_norm
                new_lb[:, 2] = yc_norm
                new_lb[:, 3] = w_norm
                new_lb[:, 4] = h_norm
                new_lb[:, 5] = lb[:, 5] # occlevel

                lb = new_lb # 변환된 레이블로 대체

                # 이제 lb는 (cls, xc_norm, yc_norm, w_norm, h_norm, occlevel) 형태입니다.
                # 이 값들은 이미 정규화되어 있어야 합니다 (0~1 사이).
                # 범위 검증을 다시 수행합니다.
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                # 작은 오차 허용 (float 연산 문제)
                assert (lb[:, 1:5] <= 1.0 + 1e-6).all() and (lb[:, 1:5] >= 0.0 - 1e-6).all(), \
                    f"non-normalized or out of bounds coordinates {lb[:, 1:5][(lb[:, 1:5] > 1.0 + 1e-6) | (lb[:, 1:5] < 0.0 - 1e-6)]} in {lb_file}"
                
                # 중복 레이블 제거 (cls, xc, yc, w, h, occlevel 모두 동일한 경우)
                # occlevel까지 포함하여 중복 검사를 하므로, occlevel이 다르면 다른 레이블로 간주됩니다.
                # 이것이 의도한 바라면 그대로 유지하고, cls+bbox만으로 중복을 제거하려면 lb[:, :5]를 사용하세요.
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    msg = f"{prefix}WARNING ⚠️ {lb_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1 # 레이블 파일은 있지만 내용이 비어있음
                lb = np.zeros((0, 6), dtype=np.float32)
        else:
            nm = 1 # 레이블 파일 자체가 없음
            lb = np.zeros((0, 6), dtype=np.float32)

        # segments는 RGBT 데이터셋에서 일반적으로 사용하지 않지만, 기본 반환값에 포함
        # 여기서는 segments를 비워둡니다. 필요하다면 따로 파싱 로직을 추가해야 합니다.
        segments = []

        return im_file, lb, (actual_img_w, actual_img_h), segments, nm, nf, ne, nc, msg

    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file.format(modalities[0])} or {lb_file}: ignoring corrupt image/label: {e}"
        # 예외 발생 시, 올바른 반환 형식을 유지하도록 합니다.
        return None, np.zeros((0, 6), dtype=np.float32), (0, 0), [], nm, nf, ne, nc, msg


def verify_rgbt_image_label(modalities, args):
    """Verifies a single image-label pair, ensuring image format, size, and legal label values."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []
    try:
        # 이미지의 실제 해상도 가져오기 (W, H)
        im_path_for_shape = im_file.format(modalities[0])
        im_pil = Image.open(im_path_for_shape)
        im_pil.verify()
        actual_img_w, actual_img_h = exif_size(im_pil) 
        
        # LOGGER.info(f"DEBUG: verify_rgbt_image_label - Image: {os.path.basename(im_path_for_shape)}, ACTUAL RAW Image W,H: ({actual_img_w}, {actual_img_h})")

        for modality in modalities:
            im = Image.open(im_file.format(modality))
            im.verify()
            shape = exif_size(im) # (W, H)
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file.format(modality), "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":
                        ImageOps.exif_transpose(Image.open(im_file.format(modality))).save(im_file.format(modality), "JPEG", subsampling=0, quality=100)
                        msg = f"{prefix}WARNING ⚠️ {im_file.format(modality)}: corrupt JPEG restored and saved"

        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                # KAIST dataset labels: N x 6 (cls, x_lefttop, y_lefttop, width, height, occlevel)
                assert lb.shape[1] == 6, f"labels require 6 columns, {lb.shape[1]} columns detected"

                # DEBUG: 변환 전 (원본 TXT에서 읽어온) 레이블 확인
                # LOGGER.info(f"DEBUG: verify_rgbt_image_label - Raw labels (from txt) for {os.path.basename(im_file)}: {lb}")

                # ===== START: KAIST 레이블 (x_lefttop, y_lefttop, width, height)을 YOLOv5 (xc, yc, w, h)로 변환 =====
                # 레이블 값들이 이미 정규화되어 있다고 가정 (0.0 ~ 1.0 사이 값)
                # Raw labels: [[ 3 0.83281 0.42773 0.032813 0.095703 1]] 이런 형태이므로
                # x_lefttop_norm, y_lefttop_norm, width_norm, height_norm 입니다.
                
                # xc = x_lefttop + width / 2
                # yc = y_lefttop + height / 2
                
                xc_norm = lb[:, 1] + lb[:, 3] / 2
                yc_norm = lb[:, 2] + lb[:, 4] / 2
                w_norm = lb[:, 3]
                h_norm = lb[:, 4]

                # 새로운 레이블을 만듭니다: (cls, xc_norm, yc_norm, w_norm, h_norm, occlevel)
                new_lb = np.zeros_like(lb)
                new_lb[:, 0] = lb[:, 0] # class_id
                new_lb[:, 1] = xc_norm
                new_lb[:, 2] = yc_norm
                new_lb[:, 3] = w_norm
                new_lb[:, 4] = h_norm
                new_lb[:, 5] = lb[:, 5] # occlevel

                lb = new_lb # 변환된 레이블로 대체

                # DEBUG: 변환 후 YOLOv5 형식으로 변환된 정규화 레이블 확인
                # LOGGER.info(f"DEBUG: verify_rgbt_image_label - Converted YOLOv5 normalized labels for {os.path.basename(im_file)}: {lb}")

                # ===== END: KAIST 레이블 변환 =====

                # 이제 lb는 (cls, xc_norm, yc_norm, w_norm, h_norm, occlevel) 형태입니다.
                # 이 값들은 이미 정규화되어 있어야 합니다 (0~1 사이).
                # 범위 검증을 다시 수행합니다.
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:5] <= 1.0 + 1e-6).all() and (lb[:, 1:5] >= 0.0 - 1e-6).all(), \
                    f"non-normalized or out of bounds coordinates {lb[:, 1:5][(lb[:, 1:5] > 1) | (lb[:, 1:5] < 0)]}"
                
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1
                lb = np.zeros((0, 6), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 6), dtype=np.float32)

        return im_file, lb, (actual_img_w, actual_img_h), segments, nm, nf, ne, nc, msg

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
