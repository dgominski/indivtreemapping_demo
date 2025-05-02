import os
from torch.utils.tensorboard import SummaryWriter
from options.options import Options
from data import create_dataset
from models import create_model
import torch
import tqdm
from data.datahelpers import UnNormalize, AverageMeter
from data.folder_dataset import FolderDataset
import itertools
from functools import reduce
from operator import mul
from util.evaluate import DictAverager, evaluate
import numpy as np
from data.datahelpers import custom_collate


HPARAMS_SWEEP = {
            "threshold": np.arange(0.01, 0.6, 0.01),
            "min_distance": range(1, 13, 2),
        }
GAMMAS = [5, 10, 20] # maximum allowed distance in pixel for a match


def test(testdataset, model, threshold, min_distance):
    averagers = [DictAverager() for _ in range(len(GAMMAS))]

    model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)
        for i, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):
            model.set_input(data)
            model.forward()
            preds = model.heatmap.detach().cpu().numpy()[:, 0] * model.valid.detach().cpu().numpy()[:, 0]
            centers = [np.array(c) if c[0] is not None else np.empty([0, 2]) for c in data["centers"]]
            for i, gamma in enumerate(GAMMAS):
                measures = evaluate(centers, preds, min_distance=min_distance, threshold_rel=None, threshold_abs=threshold, max_distance=gamma)
                averagers[i].update(measures)

    return [a.get_avg() for a in averagers]


def extract_heatmaps(valdataset, model):
    model.eval()
    with torch.no_grad():
        val_loader = valdataset.get_loader()

        heatmaps = []
        centers = []
        valids = []
        for i, data in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc='Extracting heatmaps'):
            model.set_input(data)
            model.forward()
            heatmaps.append(model.heatmap.detach().cpu().numpy()[:, 0])
            centers.append([np.array(c) if c[0] is not None else np.empty([0, 2]) for c in data["centers"]])
            valids.append(model.valid.detach().cpu().numpy()[:, 0])

    return heatmaps, centers, valids


def val(heatmaps, centers, valids, threshold, min_distance):
    averagers = [DictAverager() for _ in range(len(GAMMAS))]

    for hm_batch, centers_batch, valid_batch in zip(heatmaps, centers, valids):
        # filter out invalid pixels
        preds = hm_batch * valid_batch
        for i, gamma in enumerate(GAMMAS):
            measures = evaluate(centers_batch, preds, min_distance=min_distance, threshold_rel=None, threshold_abs=threshold, max_distance=gamma)
            averagers[i].update(measures)

    return [a.get_avg() for a in averagers]


def hparam_sweep(valdataset, model):
    best_val_f1 = 0
    best_val_measures = None
    best_hparams = None

    hparams_sweep = HPARAMS_SWEEP
    all_combinations = (dict(zip(hparams_sweep.keys(), values)) for values in itertools.product(*hparams_sweep.values()))

    # extract all heatmaps
    print('extracting heatmaps from validation set')
    heatmaps, centers, valids = extract_heatmaps(valdataset, model)
    print('looking for best hyperparameters on validation set')
    for combination in tqdm.tqdm(all_combinations, total=reduce(mul, [len(hparams_sweep[k]) for k in hparams_sweep]), desc='Hyperparameter sweep'):
        measures = val(heatmaps, centers, valids, **combination)
        val_f1 = measures[0]["fscore"]
        if val_f1 > best_val_f1:
            print(f'found better params {combination} with avg f1 @{GAMMAS[0]} = {val_f1}')
            best_val_f1 = val_f1
            best_val_measures = measures
            best_hparams = combination

    print(f'found best params {best_hparams} with avg f1 = {best_val_f1}')
    return best_val_measures, best_hparams



if __name__ == '__main__':
    opt = Options().parse()
    valdataset = FolderDataset(opt, root=opt.val_dir, mode='val')
    valdataset.load_data()
    testdataset = FolderDataset(opt, root=opt.test_dir, mode='test')
    testdataset.load_data()
    model = create_model(opt, mode='test')

    # find best hyperparameters
    best_val_measures, best_hparams = hparam_sweep(valdataset, model)
    print('Best hyperparameters:')
    print(best_hparams)
    print('######## Best validation measures:')
    for measure, gamma in zip(best_val_measures, GAMMAS):
        print(f'MAX DISTANCE (px) = {gamma}')
        print(measure)

    # test with best hyperparameters
    test_measures = test(testdataset, model, **best_hparams)
    print('######### Test measures:')
    for measure, gamma in zip(test_measures, GAMMAS):
        print(f'MAX DISTANCE (px) = {gamma}')
        print(measure)




