import numpy as np
import torch
import tqdm

from options.options import Options
from models import create_model
import os
import torchvision.transforms as transforms
import rasterio
from shapely.geometry import Point
import geopandas as gpd


def get_images_to_predict(target_dir, target_filelist=None, ext=None):
    """ Get all input images to predict

    Either takes only the images specifically listed in a text file at target_filelist,
    or all images in target_dir with the correct prefix and file type
    """
    input_images = []
    if target_filelist is not None and os.path.exists(target_filelist):
        for line in open(target_filelist):
            if os.path.isabs(line.strip()) and os.path.exists(line.strip()):      # absolute paths
                input_images.append(line.strip())
            elif os.path.exists(os.path.join(target_dir, line.strip())):          # relative paths
                input_images.append(os.path.join(target_dir, line.strip()))

        print(f"Found {len(input_images)} images in {target_filelist}.")
    else:
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if ext is not None:
                    if file.endswith(ext):
                        input_images.append(os.path.join(root, file))
                else:
                    input_images.append(os.path.join(root, file))

        print(f"Found {len(input_images)} valid images in {target_dir}.")
    if len(input_images) == 0:
        raise Exception("No images to predict.")

    return sorted(input_images)


def predict(target_fps, model, OPT):
    for fp in tqdm.tqdm(target_fps, desc="Predicting images"):
        raster = rasterio.open(fp)
        raster_array = raster.read()
        profile = raster.profile
        raster_array = raster_array.astype(float)
        raster_array = torch.from_numpy(raster_array).unsqueeze(0)

        # make tensor dimensions multiples of 2
        if raster_array.shape[2] % 2 != 0:
            raster_array = raster_array[:, :, :-1, :]
        if raster_array.shape[3] % 2 != 0:
            raster_array = raster_array[:, :, :, :-1]

        ts = []
        ts.append(transforms.Normalize(OPT.mean, OPT.std))
        transform = transforms.Compose(ts)

        with torch.no_grad():
            transformed = transform(raster_array.float())
            probs_batch = model(transformed)
            probs_batch[probs_batch > 2] = 0  # overshooting values considered false positives
            probs = probs_batch.float().cpu()
            if not OPT.only_hm:
                dets = model.decode()[0]

        output_fp = os.path.join(OPT.output_dir, os.path.splitext(os.path.split(fp)[1])[0] + ".tif")

        probs = torch.clamp(probs, min=0, max=1)[0]

        if not OPT.only_hm:
            if len(dets) > 0:
                points = [Point(rasterio.transform.xy(raster.transform, p[0], p[1])) for p in dets]
            else:
                points = []
            gdf = gpd.GeoDataFrame({'geometry': points}, crs=profile["crs"])
            schema = {'geometry': 'Point'}
            gdf.to_file(os.path.join(OPT.output_dir, os.path.splitext(os.path.split(fp)[1])[0] + ".gpkg"), index=False, schema=schema)

        new_array = probs.numpy()
        new_array = (new_array * 255).astype(int)
        profile.update(dtype="uint8", count=new_array.shape[0], compress="LZW", nodata=None, driver="GTiff")

        with rasterio.open(output_fp, 'w', **profile) as out_ds:
            out_ds.write(new_array)


if __name__ == '__main__':
    TILE_MAX_SIZE = 20000
    num_readers = 1
    loaded_queue_maxsize = 1
    num_writers = 2
    writing_queue_maxsize = 1

    OPT = Options().parse()  # get options
    if len(OPT.mean) == 0:
        raise ValueError(f"Please provide a comma separated list of mean values for patch normalization. Example: --mean 101.3,121.9,119.1,158.9")
    if len(OPT.std) == 0:
        raise ValueError(f"Please provide a comma separated list of std values for patch normalization. Example: --std 46.8,49.0,56.9,48.4")

    model = create_model(OPT, mode="test")  # create a model given opt.model and other options

    target_images = get_images_to_predict(OPT.target_dir, ext=OPT.ext)
    if not os.path.exists(OPT.output_dir):
        os.makedirs(OPT.output_dir)

    predict(target_images, model, OPT)