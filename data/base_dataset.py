"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import glob
import sys
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import tqdm
import os
from decimal import Decimal
import torch
from tqdm import tqdm
import rasterio
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.ndimage import binary_dilation
import skimage.draw
import json
from data.data_utils import draw_gaussian_windowed
from data.datahelpers import custom_collate
import matplotlib.pyplot as plt


class BaseDataset(data.Dataset, ABC):
    """
    Base class for loading datasets of geoimages.
    There are different ways of processing raw satellite images to produce ready-to-use patches, in this class we use the following definitions:
    - Raw images / rasters (self.source_paths): the images as they are delivered from the satellite imagery service/company, usually very big.
     Only a fraction of them are used in the annotation process, these are associated with a polygon (annotations) and a rectangle (area) shapefile
    - Source images images (self.source_paths): the source images processed to extract only pixels inside the rectangles (areas), associated with segmentation masks
    - Frames (self.frame_paths for big datasets, self.frame_paths & self.frames for datasets fitting in memory): the images containing all bands of interest and segmentation masks, ready to be cropped and fed to the model
    """
    def __init__(self, opt):
        self.transform = transforms.Compose((
            transforms.Resize(opt.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )) # Imagenet mean and std

        self.annotation_channel = None # Index of the annotation mask in each frame

        self.bands = []
        self.bandnames = []

        self.mode = "train"
        self.size = 0
        self.name = type(self).__name__.split('Dataset')[0]
        self.total_pixels = 0
        self.imsize = opt.imsize
        self.opt = opt

        self.sigma = 1.0 # sigma value for gaussian heatmap for tree detection

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return

    def __repr__(self):
        return

    @abstractmethod
    def load_data(self):
        # Mandatory step of either loading all the image paths or directly the images
        # self.source_paths and self.frame_paths specifically should be declared here
        return

    def get_loader(self):
        if self.mode in ["train", "val"]:
            return torch.utils.data.DataLoader(
                    self, batch_size=self.opt.batch_size, shuffle=True, num_workers=int(self.opt.num_threads), pin_memory=True, collate_fn=custom_collate, drop_last=True
                )
        else:
            return torch.utils.data.DataLoader(
                    self, batch_size=self.opt.batch_size, shuffle=False, num_workers=int(self.opt.num_threads), pin_memory=False, collate_fn=custom_collate
                )

    def print_global_stats(self):
        print(f"##### Dataset {self.name} statistics #####")
        print(f"Sources images (annotated areas): {len(self.source_paths)}")
        print(f"Frames (big images split for training and eval): {len(self.frame_paths)}")
        positive = 0
        negative = 0
        self.total_pixels = 0
        for fp in self.frame_paths:
            frame = np.load(fp)
            annotation = frame[self.annotation_channel]
            self.total_pixels += frame.shape[1]*frame.shape[2]
            positive += np.sum(annotation)
            negative += np.sum(annotation==0)
        print(f"Total pixels: {Decimal(float(self.total_pixels)):.2E} of which {Decimal(float(positive)):.2E} class pixels and {Decimal(float(negative)):.2E} background pixels")
        print(f"-- {100*np.around(positive/self.total_pixels, decimals=2)}% class annotation ratio")

    def print_split_stats(self):
        print(f"##### Dataset {self.name}, {self.mode} split statistics #####")
        # browse patches to get distribution of tree counts
        counts = []
        for i in range(len(self)):
            frame, jsondata = self.get_frame()
            patch = self.get_patch(frame, jsondata, rescale_prob=0.)
            counts.append(len(patch["centers"]) if patch["centers"] is not None else 0)
        counts = np.array(counts)
        print(f"Tree counts: total {counts.sum()}, mean {counts.mean()}, min {counts.min()}, max {counts.max()}, std {counts.std()}")

    def check_source_images(self):
        total_source_pixels = 0
        for sp in self.source_paths:
            im = rasterio.open(sp).read()
            if im.shape[1] < 128 or im.shape[2] < 128:
                print(f"Source image {sp} is under 128x128 and thus not exploitable ")
            total_source_pixels += im.shape[1] * im.shape[2]
        print(f"{total_source_pixels - self.total_pixels} pixels = {100*np.around((total_source_pixels - self.total_pixels) / total_source_pixels, decimals=2)}% were lost from source images to frames")

    def compute_mean_std(self):
        bands = self.bandnames
        psums = {b: np.array(0.) for b in bands}
        count = 0

        for i in tqdm(range(len(self.frame_paths))):
            frame, _ = self.get_frame()
            count += frame.shape[1] * frame.shape[2]
            for j, b in enumerate(bands):
                psums[b] += np.sum(frame[j])

        band_mean = {k: v / count for k, v in psums.items()}

        psums_sq = {b: np.array(0.) for b in bands}
        psums_sq["grayscale"] = torch.tensor(0.)

        count = 0
        for i in tqdm(range(len(self.frame_paths))):
            frame, _ = self.get_frame()
            count += frame.shape[1] * frame.shape[2]
            for j, b in enumerate(bands):
                psums_sq[b] += np.sum((frame[j] - band_mean[b]) ** 2)

        band_std = {k: np.sqrt(v / count) for k, v in psums_sq.items()}

        for b in bands:
            print(f"Dataset mean {band_mean[b]} and std {band_std[b]} for band {b}")

    def compute_detection_sigma(self):
        """
        Using the information in json files (source folder), runs a sweep on the sigma parameter fixing the size of gaussians
        on the tree location heatmap and computes the IoU score between the heatmap and the real polygons
        """
        # do a logarithmic sweep, 50 points
        ious = []
        sigmas = []
        json_paths = sorted(glob.glob(os.path.join(self.source_dir, "*.json")))
        for s in np.linspace(0.5, 10, 50):
            self.sigma = s
            iou_avg = AverageMeter()
            for jp in json_paths[:15]:
                jsonfile = open(jp).read()
                jsondata = json.loads(jsonfile)
                if len(jsondata) == 0:
                    print(jp)
                    continue
                centers = np.array([d["center"] for d in jsondata])[:, [1, 0]] if isinstance(jsondata, list) else np.array(jsondata["center"])[::-1]

                # draw gaussians at tree locations
                heatmap = draw_gaussian_windowed((int((centers[:, 0].max() + 100)), int((centers[:, 1].max() + 100))), centers.astype(int), sigma=s)

                gnd = np.zeros_like(heatmap)
                # draw polygons
                for d in jsondata:
                    cr = [d["geometry"][i][1] for i in range(len(d["geometry"]))]
                    cc = [d["geometry"][i][0] for i in range(len(d["geometry"]))]
                    coords = skimage.draw.polygon(cr, cc)
                    gnd[coords] = 1

                iou = evaluate_segmentation(torch.from_numpy(heatmap).unsqueeze(0) > 0.5, torch.from_numpy(gnd).unsqueeze(0))["jaccard"]
                iou_avg.update(iou)

            print(f"sigma={s} iou={iou_avg.avg}")
            ious.append(iou_avg.avg)
            sigmas.append(s)

        plt.plot(ious, sigmas)
        plt.show()


def get_annotation_mask(annotation: np.ndarray, patch_size: int):
    """Precomputes a mask giving the pixels from which it is possible to extract a patch containing at least one annotated pixel.
    Patches should be extracted starting from the top left pixel.
    Done with morphological dilation, quite slow.
    Example in 5x5 with a patch size of 2x2:
    Input       Output
    0 0 0 0 0   1 1 1 1 0
    0 1 0 1 0   1 1 1 1 0
    0 0 0 0 1   1 0 0 1 1
    1 0 0 0 0   1 0 0 0 0
    1 0 0 0 0   1 0 0 0 0
    """
    # OLD: 2d dilation, super slow
    # dilation_kernel = np.zeros((2 * patch_size, 2 * patch_size), dtype=bool)
    # dilation_kernel[:patch_size, :patch_size] = 1
    # oldmask = binary_dilation(annotation.astype(bool), dilation_kernel)
    # NEW: 2x 1D dilation, much faster
    dilation_kernel = np.zeros((2 * patch_size, 1), dtype=bool)
    dilation_kernel[:patch_size] = 1
    mask = binary_dilation(binary_dilation(annotation.astype(bool), dilation_kernel), dilation_kernel.T)
    return mask


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
