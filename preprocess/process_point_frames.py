import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import pandas as pd
import os
import os.path
import glob
import json
import rasterio
import geopandas as gpd
from rasterio import features, windows
from rasterio import transform as rt
from shapely.geometry import Polygon, Point, box
from rasterio.merge import merge
import tqdm
from skimage.exposure import rescale_intensity
import multiprocessing as mp


def get_vectorized_annotation(gdf, transform):
    """Get the annotation as a list of shapes with geometric properties (center, area).

    Each entry in the output dictionary corresponds to an annotation polygon and has the following info:
        - center: centroid of the polygon in pixel coordinates
    """

    # Invert to get transform from real world coordinates to pixel coordinates
    transform = ~transform
    trsfrm = [element for tupl in transform.column_vectors for element in tupl]
    gdf["geometry"] = gdf["geometry"].affine_transform(trsfrm[:6])
    gdf["center"] = gdf.centroid
    gdf["center"] = gdf["center"].apply(lambda p: (p.x, p.y))

    # Convert dataframe to dict
    polygons = pd.DataFrame(gdf)
    polygons.drop(labels=["geometry"], inplace=True, axis=1)
    dic = polygons.to_dict(orient="records")
    return dic


def preprocess_all(args):
    rasters = glob.glob(os.path.join(args.raster_dir, f"*.{args.raster_ext}"))
    rasterfiles = np.array(rasters)

    shp = gpd.read_file(args.point_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    footprints = []
    for rasterpath in rasters:
        raster = rasterio.open(rasterpath)
        footprints.append(raster.bounds)

    # get extent of rasters
    raster_polys = [box(*b) for b in footprints]

    rectangles = gpd.read_file(args.rectangle_file)
    rectangles = rectangles[~rectangles.is_empty]

    # browse once to get bigger rectangles
    reading_bounds = []
    without_raster = 0
    for index, row in tqdm.tqdm(rectangles.iterrows(), desc="Browsing database to generate bounds with target frame size", total=len(rectangles.index)):  # Looping over all points
        geom = row["geometry"]
        bounds = geom.bounds
        annotation_zone_polygon = box(*bounds)

        intersects = np.array([annotation_zone_polygon.intersects(poly) for poly in raster_polys])
        if intersects.sum() == 0:
            print(f"Did not find any raster for annotation zone {bounds} (CRS {rectangles.crs})")
            without_raster += 1
            continue
        rst = rasterio.open(rasterfiles[intersects][0])
        window = windows.from_bounds(*geom.bounds, transform=rst.transform)
        windowparams = [window.col_off, window.row_off, window.width, window.height]
        if window.height < args.min_frame_size:
            diff = args.min_frame_size - window.height
            windowparams[1] = window.row_off - diff / 2
            windowparams[3] = args.min_frame_size
        if window.width < args.min_frame_size:
            diff = args.min_frame_size - window.width
            windowparams[0] = window.col_off - diff / 2
            windowparams[2] = args.min_frame_size
        window = windows.Window(*windowparams)
        reading_bounds.append(windows.bounds(window, transform=rst.transform))

    print(f"{without_raster}/{len(rectangles.index)} annotation zones does not have any corresponding raster(s)")

    indices = range(len(reading_bounds))
    with mp.Pool(args.num_workers) as pool:
        pool.starmap(rasterize_and_save, [(rasterfiles, rectangles, args, i, raster_polys, shp, b) for i, b in tqdm.tqdm(zip(indices, reading_bounds), desc="Processing annotation zones", total=len(reading_bounds))])


def rasterize_and_save(rasterfiles, rectangles, args, i, raster_polys, shp, b):
    p = Polygon.from_bounds(*b)
    # find intersecting rasters
    intersects = np.array([p.intersects(poly) for poly in raster_polys])

    # TODO: add boundless=True when reading and check in relevant cases if the downstream arrays are correct
    if sum(intersects) == 0:
        raise ValueError("annotation zone does not fall inside any raster")
    elif sum(intersects) == 1:
        rst = rasterio.open(rasterfiles[intersects].item())
        window = windows.from_bounds(*p.bounds, transform=rst.transform)
        npraster = rst.read(window=window)
    elif sum(intersects) > 1:
        to_merge = [rasterio.open(r, mode="r") for r in rasterfiles[intersects].tolist()]
        npraster, _ = merge(to_merge, bounds=b)
    else:
        raise ValueError()

    transform = rt.from_bounds(p.bounds[0], p.bounds[1], p.bounds[2], p.bounds[3], npraster.shape[2], npraster.shape[1])

    validity_band = 0 * np.ones(npraster.shape[1:])
    try:
        validity_rectangles = (geom for geom in rectangles.geometry)
        validity_mask = \
        features.rasterize(shapes=validity_rectangles, default_value=1, fill=-1, out=validity_band, transform=transform,
                           dtype=rasterio.int16, all_touched=True)[None,]

    except ValueError as ve:
        print(f"got error {ve} when rasterizing")
        return

    if args.float:
        npraster = npraster.astype(np.float32)
        npraster = rescale_intensity(npraster, in_range=(0, 10000), out_range=(0, 255))

    out_array = np.concatenate((npraster, validity_mask))

    # convert polygon annotations to json
    isinarea = shp[shp.within(box(*p.bounds))].explode(column="geometry", index_parts=False)
    dic = get_vectorized_annotation(isinarea, transform)

    output_fp = os.path.join(args.output_dir, f"{i}.json")
    with open(output_fp, 'w') as fp:
        json.dump(dic, fp)

    output_fp = os.path.join(args.output_dir, f"{i}.npy")
    np.save(output_fp, out_array)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert rasters and annotation files into DL-ready patches')
    parser.add_argument('--raster-dir', type=str, required=True, help='target directory with preprocessed rasters')
    parser.add_argument('--output-dir', type=str, required=True, help='output directory to save patches')
    parser.add_argument('--point-file', type=str, required=True, help='point annotation file')
    parser.add_argument('--rectangle-file', type=str, required=True, help='rectangle annotation zones file')
    parser.add_argument('--min-frame-size', type=int, required=True, help='minimum frame size')
    parser.add_argument('--crs', type=str, required=True, default="25832", help='equal area CRS to use')
    parser.add_argument('--raster-ext', type=str, required=True, help='raster files extension')
    parser.add_argument('--float', action='store_true', help='save as float32')
    parser.add_argument('--num-workers', type=int, required=False, default=1, help='number of workers to use')
    args = parser.parse_args()
    preprocess_all(args)