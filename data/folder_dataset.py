import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np
import tqdm
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import glob
from data import create_dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import json
from data.data_utils import draw_gaussian_windowed


def custom_cropper(image, valid, keypoints, imsize):
    # crop without albumentaions
    frame = torch.from_numpy(image)
    valid = torch.from_numpy(valid)
    keypoints = torch.Tensor(keypoints)
    # random crop
    h, w = frame.shape[:2]
    if h < imsize or w < imsize:
        raise ValueError("Frame is smaller than imsize")
    top = np.random.randint(0, h - imsize)
    left = np.random.randint(0, w - imsize)
    bottom = top + imsize
    right = left + imsize
    frame = frame[top:bottom, left:right, :]
    valid = valid[top:bottom, left:right]
    if keypoints.shape[0] == 0:
        keypoints = np.empty((0, 2))
    else:
        keypoints = keypoints - np.array([left, top])
        keypoints = keypoints[keypoints[:, 0] >= 0]
        keypoints = keypoints[keypoints[:, 1] >= 0]
        keypoints = keypoints[keypoints[:, 0] < imsize]
        keypoints = keypoints[keypoints[:, 1] < imsize]
    frame = frame.permute(2, 0, 1)
    return {"image": frame, "valid": valid, "keypoints": keypoints.tolist()}


class FolderDataset(BaseDataset):
    """
    Input images band order : [RED, GREEN, BLUE, INFRARED, VALIDITY MASK]
    """
    def __init__(self, opt, root, mode="train"):

        super().__init__(opt)

        self.valid_channel = 4
        self.ground_resolution = 3.

        self.frame_dir = root

        self.normalizer = transforms.Normalize((101.3, 121.9, 119.1, 158.9),
                                 (46.8, 49.0, 56.9, 48.4))

        if mode == "val":
            # random crop
            # self.cropper = A.Compose([
            #     A.PadIfNeeded(min_height=opt.imsize, min_width=opt.imsize, border_mode=cv2.BORDER_CONSTANT),
            #     A.RandomCrop(width=opt.imsize, height=opt.imsize),
            # ], keypoint_params=A.KeypointParams(format='yx'),
            #     additional_targets={
            #         'valid': 'image',
            #         }
            # )
            self.cropper = custom_cropper

        if mode == "test":
            # no crop
            self.cropper = A.Compose([],)

        self.totensor = ToTensorV2()

    def __getitem__(self, i):
        frame = self.get_frame(from_memory=self.opt.preload)

        if frame["centers"] is None:
            frame["centers"] = []
        else:
            frame["centers"] = frame["centers"].tolist()

        cropped = self.cropper(image=frame["image"].transpose(1, 2, 0),
                               keypoints=frame["centers"],
                               valid=frame["valid"], imsize=self.opt.imsize)

        # cropped = self.totensor(**cropped)

        im = self.normalizer(cropped["image"].float())

        # Fill in sampledict with selected bands
        sampledict = {}
        sampledict["input"] = im
        sampledict["valid"] = cropped["valid"].unsqueeze(0)
        sampledict["rawinput"] = cropped["image"].permute(2, 0, 1)[:3] / 255
        sampledict["centers"] = cropped["keypoints"] if len(cropped["keypoints"]) > 0 else [None]
        sampledict["resolution"] = self.ground_resolution

        return sampledict

    def get_frame(self, from_memory=False):
        # randomly pick frame
        picked = np.random.randint(len(self.frame_paths))
        if from_memory:
            frame = self.frames[picked]
        else:
            frame = self.frame_paths[picked]
            frame = np.load(frame)
        # make frame dimensions multiple of 2
        if frame.shape[1] % 2 != 0:
            frame = np.pad(frame, ((0, 0), (0, 1), (0, 0)), 'edge')
        if frame.shape[2] % 2 != 0:
            frame = np.pad(frame, ((0, 0), (0, 0), (0, 1)), 'edge')

        jsonfile = open(self.json_paths[picked]).read()
        jsondata = json.loads(jsonfile)

        image = np.delete(frame, [self.valid_channel], axis=0)
        valid = frame[self.valid_channel]

        if len(jsondata) == 0:
            centers = None
        else:
            # Get centers
            centers = np.array([d["center"] for d in jsondata])[:, [1, 0]]

        return {"image": image.astype(np.uint8),
                "valid": valid,
                "centers": centers}

    def load_data(self):
        self.frames = []
        self.json_paths = []
        self.frame_paths = []
        all_frame_paths = sorted(glob.glob(os.path.join(self.frame_dir, "*.npy")))
        for f in all_frame_paths:
            self.json_paths.append(os.path.splitext(f)[0] + ".json")
            self.frame_paths.append(f)
        if len(self.json_paths) != len(self.frame_paths):
            raise ValueError("Frame and json files number not matching !")
        return

    def __len__(self):
        return self.opt.epoch_size
