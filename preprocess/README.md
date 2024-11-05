## Individual Tree Mapping (PyTorch)

Data preprocessing utilities.

---
 
### Why?

The training and prediction pipelines are designed to work with preprocessed data, meaning:
* The source rasters are limited in size, so that they can fit in memory.
* The source rasters' pixel values are limited to a certain range, with outliers filtered out and a relatively even distribution of values.
* The annotation vector files are associated with individual frames so that they can be easily loaded and processed.

Here you can find example scripts for preprocessing PlanetScope/RapidEye rasters and annotations.
They apply scaling, outlier removal and histogram equalization (using skimage's CLAHE algorithm), and output uInt8 rasters with associated json files.
This has been found to work well on specific studies, but you may need to adapt the preprocessing to your own data.

---

### Raster preprocessing:

All raster preprocessing scripts use multiprocessing to speed up computation and I/O operations. This is implemented with dedicated
reading, processing and writing processes, and queues to pass data between them. By default, the number of processes
and the size of all queues is set to 1. If you have a lot of resources, feel free to increase those numbers, they are hardcoded (variables num_readers, num_processors, num_writers
and Queue(maxsize=N)). 

OUTLIER_PERCENTAGE defines the percentage of values in the low and high tails of the distribution that will be clipped.

CLIP_LIMIT defines the contrast limit for each histogram bin in the CLAHE algorithm. [See here for a comprehensive explanation.](https://amroamroamro.github.io/mexopencv/opencv/clahe_demo_gui.html)

MAXIMUM_RASTER_SIZE defines the maximum size of the output rasters. If the input rasters are larger, they will be split into the minimal number of rasters such
that no raster is bigger that this.

* `preprocess_rapideye_rasters.py`: preprocesses RapidEye rasters (5m ground resolution). 
* `preprocess_planetscope_rasters.py`: preprocesses PlanetScope rasters (3m ground resolution). 

---

### Annotation preprocessing:

Annotation preprocessing scripts take as inputs a folder with preprocessed rasters, a vector file with point or polygon annotations, and a vector file with rectangles indicating what zones were annotated.
It generates frames by cropping the rasters to the labeled rectangles, and saves the annotations as json files.

**Features**:

_--min-frame-size_ defines the minimal frame size in pixels. If the labeled rectangle is smaller than this, it will be extended to this size, but pixels outside of the annotation rectangle will be considered invalid and not used for computing the loss.

The script handles annotation rectangles covering multiple rasters (using rasterio.merge).

 ⚠️ There might be issues if there is a mismatch between the CRS of the rasters and the CRS of the annotation files. If you encounter this, please convert them to the same CRS, 
or uncomment the lines converting geometries to a common provided --crs 

---

### Example script:

```bash
python3 -m preprocess.process_point_frames.py --raster-dir .../rasters/ --output-dir .../frames/ --point-file .../points.gpkg --rectangle-file ../rectangles.gpkg --min-frame-size 512 --crs 4326 --raster-ext tif
```
