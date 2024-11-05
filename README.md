# Individual Tree Mapping (PyTorch)

Training and prediction scripts for individual tree mapping.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Project structure](#project-structure)
3. [Datasets](#datasets)
4. [Models](#models)
5. [Preprocessing](#preprocessing)
6. [Prediction](#prediction)
7. [Merging redundant predictions](#merging-redundant-predictions)
8. [Re-training with new samples](#re-training-with-new-samples)
9. [Test-time augmentation](#test-time-augmentation)
10. [TL;DR](#ready-to-go-routine--tldr)

### Dependencies:

You can use the provided .yml Conda environment file to easily install dependencies into a separate environment:
```
# From repo root
conda env create -f environment.yml OR mamba env create -f environment.yml
conda activate treemapping
```

Check out [mamba](https://mamba.readthedocs.io/en/latest/index.html) for super-fast environment creation.

---
 
### Project structure:

The folder <b>data</b> contains one Python file for each dataset. The class in that file defines how to load the data, what data augmentation to apply, prints statistics..

The folder <b>models</b> contains one Python file for each model. The class in that file defines the model architecture, the loss function, the optimizer, the forward and backward passes..

Have a look at options/options.py to see all the available command-line arguments.

---
 
### Datasets:

One dataset is defined by a set of images and labels.
The following datasets are currently implemented:
* PlanetScope India (indiaplanetscope.py): individual point tree labels made by Martin, out-of-forest trees, ~3m ground resolution
* RapidEye India (indiarapideye.py): individual point tree labels made by Martin, out-of-forest trees, ~5m ground resolution

The current data logic is hierarchical. Source **rasters** are preprocessed separately to build DL-ready **frames**. 
During training and validation, frames are loaded from the disk and randomly cropped into training **patches**, with optional data augmentation.
The *[Albumentations](https://albumentations.ai/)* library is very useful to make this quick and easy. 
Patches are associated with all the necessary labels and metadata (e.g. validity mask when the patch is not entirely annotated, ground resolution) in a dictionary. The dataloader automatically merges dictionaries in training batches.

<details>
<summary><b>Creating your own dataset</b></summary><br/>
To create your own dataset, you can start from one of the dataset files and modify:

1. load_data() to load all image and label paths (or directly the images if you have a lot of RAM). Root directories are hardcoded.
2. get_frame() to load the frame from disk.
3. \_\_getitem__(self, i) to build the training dictionary. 

The BaseDataset class contains some useful functions and helps structuring the dataset classes.

</details>

---

### Models:

One model is defined by an architecture and a set of functions to optimize parameters and infer predictions.

The following models are currently implemented:
* Masked Gaussian Detection (maskedgdet.py): individual tree detection with point labels, using adaptive heatmaps and supporting masked targets (with the validity mask).
* Ensemble Model (multigdet.py): ensemble model, used for prediction to load mulitple GDet checkpoints and average predictions.

<details>
<summary><b>Recommended reads for understanding the technical choices </b></summary><br/>

Adaptive heatmaps: Z. Luo, Z. Wang, Y. Huang, L. Wang, T. Tan, and E. Zhou, [“Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation,”](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Rethinking_the_Heatmap_Regression_for_Bottom-Up_Human_Pose_Estimation_CVPR_2021_paper.pdf) in CVPR, 2021.

Point-based tree detection: Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery https://arxiv.org/abs/2208.10607

Point-based detection: X. Zhou, D. Wang, and P. Krähenbühl, “Objects as Points,” arXiv, arXiv:1904.07850, Apr. 2019. doi: 10.48550/arXiv.1904.07850.

</details>

<details>
<summary><b>Should I start from a pretrained model?</b></summary><br/>

There is [extensive](https://arxiv.org/abs/1902.06148) [evidence](https://arxiv.org/abs/1901.09960) that [pretraining](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf) [on a large dataset](https://arxiv.org/abs/2104.10972) helps getting better models, especially when re-training on a small dataset.
Common pretraining datasets include ImageNet, COCO, Places365, and in the context of remote sensing, BigEarthNet (Sentinel-2). 
It might seem conterintuitive given the very different images and tasks, but it turns out that the features learned on large-scale datasets generalize well to new tasks, and are just overall better than random initialization.

This being said, it is not always clear what pretrained model to use, and how to adapt it to your task. Here we propose to use an encoder pretrained on BigEarthNet, and train the decoder from scratch. The first layer of the model can be modified to adapt to the number of channels in your input data.

When starting from a pretrained model, it is important to use a low learning rate to avoid erasing previous knowledge.
</details>

<details>
<summary><b>How can the model estimate size only from points?</b></summary><br/>

In [“Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation,”](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Rethinking_the_Heatmap_Regression_for_Bottom-Up_Human_Pose_Estimation_CVPR_2021_paper.pdf), authors propose a model that adaptively rescales the target gaussians during training. The model outputs a heatmap and a size map, the size map is used to scale the gaussians positioned at the labeled centers. This is implemented in the MaskedGDet model.

The underlying assumption is that the bigger the object is on the image, the higher the chance of the labeled center being shifted from the actual center. During training, the model can adapt the gaussian size to produce large gaussians for bigger objects, so that the gaussian aligns better with visual features.
Therefore, given enough samples with varying sizes and a certain level of noise in point labels, we assume that the size map correlates somehow with actual object sizes.

Of course this is far from perfect. Uncertainty for tree positions can also depend on many other factors, including tree species, image quality, and annotation quality.

To tune this, two parameters are available:
- `--delta` tunes the regularization of the uncertainty map. High values => no tolerance => enforcing small, confident gaussians (can lead to suboptimal models). Low values => high tolerance => letting the model tune gaussian size (can lead to overfitting). 
- `--sigma` tunes the initial size of the gaussians, ie the upper limit of detail that the model can reach (smaller gaussians = precise models, more difficult to optimize).

</details>

---

### Preprocessing:

The preprocessing folder is used separately from the rest of the project, to prepare the data for training and prediction.
See the README in that folder for more details.

---

### Prediction:

The prediction script uses multiprocessing to speed up computation and I/O operations. This is implemented with dedicated
reading, processing and writing processes, and queues to pass data between them. By default, the number of processes
and the size of all queues is set to 1. If you have a lot of resources, feel free to increase those numbers, they are hardcoded (variables num_readers, loaded_queue_maxsize, 
num_writers, writing_queue_maxsize).  

A summary of important parameters is printed when the script is launched. In particular, you can adjust the following:

`--imsize N` size of the patches fed to the network. Does not have to be the same as during training, usually bigger is better to avoid border effects.

`--overlap N` overlap in pixels between patches when predicting. This is useful to avoid artifacts at the edges of the patches.

`--threshold t` threshold to detect positives on the heatmap. Lower threshold = more predictions and more false positives, higher threshold = less predictions and more false negatives.

`--nms-kernel-size N` minimum spacing in pixels between two predictions. Must be odd.


Have a look at the GaussianHeatmapDecoder in predict.py for other parameters for estimating size from the spread of gaussian blobs.

:heavy_exclamation_mark: Default values for the threshold and NMS kernel size are (0.35, 3). Those values were found with a sweep on an independent validation set on PlanetScope images in India. They are likely to be suboptimal for other datasets.

---

### Merging redundant predictions:

If you have access to multiple images on the same area at different dates and can assume that there is no significant change, you can merge the predictions to get a more robust result.

Example:
`python3 merge_multiple_dates.py --target-dir $TARGET_DIR --output-dir $OUTPUT_DIR --gnd 3 --batch-size 8`

---

### Re-training with new samples:

If you want to retrain an existing model with new samples without retraining from scratch, use data/updated_dataset.py. Modify the frame directories directly in the code + the sampling weight. The weight gives the frequency at which new frames are loaded. It is recommended to retrain with a lower learning rate.

Example:
`python3 train_centers_only.py --model maskedgdet --load-from ../checkpoints/CKPT_NAME --train-dataset updated --batch-size 8 --imsize 384 --nbands 4 --gpu-ids 0 --lr 5e-7 --print-freq 100 --checkpoints-dir ../tree_mapping_example_data/ckpts --sigma 4.5 --nepoch 500 --num-threads 16 --delta 0.01 --no-val`

:heavy_exclamation_mark: Validation is not supported yet with the updated dataset. 

---

### Test-time augmentation:

A simple trick to boost performance is to merge predictions done on the original image and on augmented versions of it (by default rotation, flipping, rescaling). This is implemented in the prediction pipeline with the `--tta` flag. The predictions are flipped back before merging. Feel free to code different transformations, but note that it can severely slow down the prediction.

---

## Ready-to-go routine / TL;DR:

The routine below describes the full preprocessing, training and prediction pipeline. Skip to the bottom if you only want to predict!

Copy the folder in 10.61.69.20/Users/Dimitri/tree_mapping_example_data on your local machine.

1. Preprocess the rasters.

`python3 -m preprocess.process_planetscope_rasters --target-dir ../tree_mapping_example_data/source_rasters --output-dir ../tree_mapping_example_data/preprocessed_rasters --ext jp2`

2. Create the frames.

`python3 -m preprocess.process_fixed_size_points --raster-dir ../tree_mapping_example_data/preprocessed_rasters --output-dir ../tree_mapping_example_data/frames --point-file ../tree_mapping_example_data/annotation_files/points.shp --rectangle-file ../tree_mapping_example_data/annotation_files/rectangles.shp --patch-size 512 --crs 32743 --raster-ext jp2`

3. Divide the frames into train/val splits. You can do this manually, or use the train_val_split method: modify the hardcoded dataroot in data.template_dataset.py then run `python3 -m data.template_dataset`
4. Fill in mean and std values for the normalizer in data.template_dataset.py
5. Train! 

`python3 train_centers_only.py --model maskedgdet --train-dataset template --batch-size 8 --imsize 384 --nbands 4 --gpu-ids 0 --lr 1e-5 --val-freq 5 --print-freq 100 --checkpoints-dir ../tree_mapping_example_data/ckpts --sigma 4.5 --nepoch 500 --num-threads 16 --delta 0.01
`

6. Monitor the training process with Tensorboard: `tensorboard --logdir ../tree_mapping_example_data/ckpts` 
7. After a while you should see predictions appear and the F1 detection rate go up. When the validation loss and F1 are stabilized, it's time to predict! Don't forget to change the checkpoint name in the command below (CKPT_NAME, e.g. ckpts/0404_1556) and update mean and std values.

`python3 predict.py --model maskedgdet --load-from ../tree_mapping_example_data/ckpts/CKPT_NAME --target-dir ../tree_mapping_example_data/preprocessed_rasters/ --output-dir ../tree_mapping_example_data/predictions --pick-bands 0,1,2,3 --mean 101.3,121.9,119.1,158.9 --std 46.8,49.0,56.9,48.4 --gnd 3. --nbands 4 --imsize 1024 --batch-size 4 ` 

**If you only want to run predictions**, you can skip steps 2-6 and directly predict with one of the pretrained models in 10.61.69.20/Users/Dimitri/checkpoints/gdet_india:

`python3 predict.py --model maskedgdet --load-from ../Users/Dimitri/checkpoints/gdet_india/CKPT_NAME --target-dir $TARGET_DIR --output-dir $OUTPUT_DIR --pick-bands 0,1,2,3 --mean 101.3,121.9,119.1,158.9 --std 46.8,49.0,56.9,48.4 --gnd 3. --nbands 4 --imsize 1024 --batch-size 4 ` 
