# Individual Tree Mapping (PyTorch)

Training and prediction scripts for individual tree mapping.

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

1. Preprocess the rasters and create the frames

=> see the README in the preprocess folder
2. Divide the frames into train/val splits. You can do this manually, or use the train_val_split method: modify the hardcoded dataroot in data.template_dataset.py then run `python3 -m data.template_dataset`
3. Fill in mean and std values for the normalizer in data.template_dataset.py
4. Train! 

`python3 train.py --model gdet --train-dataset frame --batch-size 8 --imsize 384 --nbands 4 --gpu-ids 0 --lr 1e-5 --val-freq 5 --print-freq 100 --checkpoints-dir ... --sigma 4.5 --nepoch 500 --num-threads 16 --delta 0.01
`

5. Monitor the training process with Tensorboard: `tensorboard --logdir ../tree_mapping_example_data/ckpts` 
6. After a while you should see predictions appear and the F1 detection rate go up. When the validation loss and F1 are stabilized, it's time to predict! Don't forget to change the checkpoint name in the command below (CKPT_NAME, e.g. ckpts/0404_1556) and update mean and std values.

`python3 predict.py --model gdet --target-dir .../target_patches/ --output-dir .../preds --mean 101.3,121.9,119.1,158.9 --std 46.8,49.0,56.9,48.4 --imsize 1024 --batch-size 4 --nbands 4 --threshold 0.035`
