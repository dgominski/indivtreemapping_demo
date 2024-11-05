# Individual Tree Mapping (PyTorch)

Training and prediction scripts for individual tree mapping.

## Table of Contents
1. [Dependencies](#dependencies)
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

## Ready-to-go routine:

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
