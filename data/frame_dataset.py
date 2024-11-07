import os.path
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


class FrameDataset(BaseDataset):
    """
    Input images band order : [RED, GREEN, BLUE, INFRARED, VALIDITY MASK]
    """
    def __init__(self, opt, mode="train"):

        super().__init__(opt)

        self.bandnames = ["red", "green", "blue", "infrared"]
        self.mode = mode

        self.valid_channel = 4
        self.ground_resolution = 3.

        self.root = "/scratch/tmp/indivtreemaps_demo/skysat/frames/"
        self.frame_dir = os.path.join(self.root, mode)

        self.normalizer = transforms.Normalize((101.3, 121.9, 119.1, 158.9),
                                 (46.8, 49.0, 56.9, 48.4))

        self.cropper = A.Compose([
            A.RandomSizedCrop(width=opt.imsize, height=opt.imsize, min_max_height=(int(opt.imsize * 4/5), int(opt.imsize * 6/5)), interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ], keypoint_params=A.KeypointParams(format='yx'),
            additional_targets={
                'valid': 'image',
                'weightmap': 'image',}
        )
        self.augmenter = A.Compose([
            ToTensorV2(),
        ])

    def __getitem__(self, i):
        frame = self.get_frame(from_memory=self.opt.preload)

        if frame["centers"] is None:
            frame["centers"] = []
        else:
            frame["centers"] = frame["centers"].tolist()

        cropped = self.cropper(image=frame["image"].transpose(1, 2, 0),
                               keypoints=frame["centers"],
                               valid=frame["valid"],
                               weightmap=frame["weightmap"])
        transformed = self.augmenter(image=cropped["image"].astype(np.float32))

        im = self.normalizer(transformed["image"].float())

        # Fill in sampledict with selected bands
        sampledict = {}
        sampledict["input"] = im
        sampledict["valid"] = torch.from_numpy(cropped["valid"]).unsqueeze(0)
        sampledict["weightmap"] = torch.from_numpy(cropped["weightmap"]).unsqueeze(0)
        sampledict["rawinput"] = torch.from_numpy(cropped["image"].transpose(2, 0, 1)[:3]) / 255
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
        jsonfile = open(self.json_paths[picked]).read()
        jsondata = json.loads(jsonfile)

        image = np.delete(frame, [self.valid_channel], axis=0)
        valid = frame[self.valid_channel]

        if len(jsondata) == 0:
            centers = None
            weightmap = np.zeros((image.shape[1], image.shape[2]))
        else:
            # Get centers
            centers = np.array([d["center"] for d in jsondata])[:, [1, 0]]

            # Draw heatmap from centers
            centers_to_draw = centers
            pseudoradiuses_to_draw = centers_to_draw.shape[0] * [self.sigma]
            weightmap = draw_gaussian_windowed((image.shape[1], image.shape[2]), centers_to_draw.astype(int), sigmas=pseudoradiuses_to_draw)

        return {"image": image.astype(np.uint8),
                "valid": valid,
                "weightmap": weightmap,
                "centers": centers}

    def load_data(self):
        self.frames = []
        self.json_paths = []
        self.frame_paths = []
        all_frame_paths = sorted(glob.glob(os.path.join(self.frame_dir, "*.npy")))
        for f in all_frame_paths:
            frame = np.load(f)
            # only keep frames bigger than imsize
            if frame.shape[1] >= self.opt.imsize and frame.shape[2] >= self.opt.imsize:
                self.json_paths.append(os.path.splitext(f)[0] + ".json")
                self.frame_paths.append(f)
                if self.opt.preload:
                    self.frames.append(frame)
        if len(self.json_paths) != len(self.frame_paths):
            raise ValueError("Frame and json files number not matching !")
        return

    def __len__(self):
        return self.opt.epoch_size


if __name__ == '__main__':
    from options.options import Options
    import seaborn as sns
    import matplotlib.pyplot as plt
    opt = Options()
    opt = opt.parse()
    data = create_dataset(opt, name=opt.train_dataset, mode='train')
    loader = data.get_loader()
    n_trees = []
    for i, data in tqdm.tqdm(enumerate(loader), total=len(loader)):  # inner loop within one epoch
        ims = data["input"]
        print(ims.shape)
        if data["centers"][0] is not None:
            n_trees.append(len(data["centers"][0]))
        else:
            n_trees.append(0)
    sns.histplot(n_trees)
    plt.show()