import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import rasterio
import numpy as np
import torch


class RasterDataset(BaseDataset):
    def __init__(self, opt, mode="train"):

        super().__init__(opt)
        self.annotation_channel = 1
        self.boundary_channel = 0
        self.raster_path = "/scratch/target_data/france_split/69-2020-0840-6525-LA93-0M20-E080_3_2.tif"
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1)

        self.normalize = transforms.Normalize((0.361, 0.3861, 0.3353),
                                 (0.177, 0.147, 0.131))

        self.ground_resolution = 0.2

    def __getitem__(self, i):
        frame = self.get_frame()

        patch = self.get_patch(frame)
        im = self.normalize(patch["image"])
        grayscale = self.grayscale_transform(im[:3])

        im, grayscale, annotation, boundaries, heatmap = [
            im, grayscale, patch["annotation"],
            patch["boundaries"],
            patch["heatmap"],
        ]
        centers = patch["centers"]
        vertices = patch["vertices"]

        # Fill in sampledict with selected bands
        sampledict = {}
        for b in self.bands:
            sampledict[self.bandnames[b]] = im[b]
        # Add grayscale
        sampledict["grayscale"] = grayscale
        # Add full image and target
        sampledict["input"] = im[:3]
        sampledict["rawinput"] = patch["image"][:3]
        sampledict["target"] = annotation
        sampledict["heatmap"] = heatmap
        sampledict["boundaries"] = boundaries
        sampledict["centers"] = centers.tolist() if centers is not None else [None]
        sampledict["pseudoradius"] = patch["pseudoradiuses"].tolist() if patch["pseudoradiuses"] is not None else [None]
        sampledict["resolution"] = self.ground_resolution / patch["scaling_factor"]
        sampledict["vertices"] = vertices if vertices is not None else [None]

        return sampledict

    def load_data(self):
        return

    def get_frame(self, from_memory=False):
        frame = rasterio.open(self.raster_path).read().astype(float) / 255
        self.current_frame_size = [(frame.shape[1] // self.imsize) - 1, (frame.shape[2] // self.imsize) - 1]
        return frame

    def get_patch(self, frame):
        # Get patch identified with incremental attributes
        image = frame[:, self.current_patch[0] * self.imsize:(self.current_patch[0] + 1) * self.imsize,
                          self.current_patch[1] * self.imsize:(self.current_patch[1] + 1) * self.imsize]
        annotation = np.zeros(image.shape[1:])
        boundaries = np.zeros(image.shape[1:])

        heatmap = np.zeros((self.imsize, self.imsize))

        centers = None
        pseudoradiuses = None
        vertices_in_patch = None

        if self.current_patch[0] == self.current_frame_size[0] and self.current_patch[1] == self.current_frame_size[1]:
            # next frame, reset row and column
            self.current_frame += 1
            self.current_patch[0] = 0
            self.current_patch[1] = 0
        elif self.current_patch[1] == self.current_frame_size[1]:
            # next row, reset column
            self.current_patch[0] += 1
            self.current_patch[1] = 0
        else:
            # next column
            self.current_patch[1] += 1

        image = torch.from_numpy(image)
        annotation = torch.from_numpy(annotation).unsqueeze(0)
        boundaries = torch.from_numpy(boundaries).unsqueeze(0)
        boundaries[boundaries > .5] = 10.
        boundaries[boundaries < .5] = 1.
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)

        return {"image": image,
                "annotation": annotation,
                "boundaries": boundaries,
                "heatmap": heatmap,
                "centers": centers,
                "pseudoradiuses": pseudoradiuses,
                "scaling_factor": 1,
                "vertices": vertices_in_patch}

    def setup(self):
        self.size = 1000
        pass


if __name__ == '__main__':
    from options.options import Options
    from data import create_dataset

    opt = Options()
    opt = opt.parse()
    valdata = create_dataset(opt, name=opt.train_dataset, mode='val')
    a = valdata[0]
    sadsad