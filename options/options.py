import argparse
import os
import torch
import models
import data
import datetime
import json
import copy
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%m%d_%H%M"), help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints-dir', default="/tmp/", type=str, help='models are saved here')
        parser.add_argument('--load-from', type=str, default=None)
        parser.add_argument('--load-FE', type=str, default=None)
        parser.add_argument('--load-epoch', type=str, default=None)
        parser.add_argument('--print-freq', type=int, default=100, help='frequency of showing training results on console')

        # model parameters
        parser.add_argument('--model', type=str, default='unet', help='chooses which model to use.')
        parser.add_argument('--framework', type=str, default=None, help='type of framework: [seg, hm, hm+sm]')
        parser.add_argument('--net', type=str, default='resnet50', help='chooses which backcbone network to use.')
        parser.add_argument('--alpha', type=float, default=1.0, help='parameter for loss weighing')
        parser.add_argument('--beta', type=float, default=1.0, help='parameter for loss weighing')
        parser.add_argument('--delta', type=float, default=1.0, help='parameter for loss weighing')
        parser.add_argument('--gamma', type=float, default=10., help='maximum distance in pixels allowed for a detection to be considered a true positive')
        parser.add_argument('--n', type=int, default=2000, help='parameter for model')
        parser.add_argument('--sigma', type=float, default=4.5, help='default sigma value for gaussian heatmaps')
        parser.add_argument('--threshold', type=float, default=.35, help='threshold parameter for heatmap decoding')
        parser.add_argument('--nms-kernel-size', type=int, default=3, help='kernel size for heatmap decoding')
        parser.add_argument('--nbands', type=int, default=3, help='number of bands to use')
        parser.add_argument('--not-pretrained', dest='pretrained', action='store_false', help='initialize model with random weights (default: pretrained on imagenet)')

        # dataset parameters
        parser.add_argument('--train-dataset', type=str, default='frame', help='training dataset name')
        parser.add_argument('--pick-bands', default='', help='comma separated list of bands : 0|1|2|3|...')
        parser.add_argument('--num-threads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--batch-size', type=int, default=8, help='input batch size')
        parser.add_argument('--load-size', type=int, default=286, help='loading image size')
        parser.add_argument('--imsize', type=int, default=256, help='final image size')
        parser.add_argument('--fixed-size-gaussian', action='store_true')
        parser.add_argument('--toy', action='store_true', help='train on toy dataset (small size)')
        parser.add_argument('--query', type=str, help='query to load from disk and feed to model')
        parser.add_argument('--ratio', type=float)
        parser.add_argument('--split', type=str, default="A", help="select train/val split")
        parser.add_argument('--gnd', type=float, default=3.0, help="ground resolution")
        parser.add_argument('--mean', default='', help='comma separated image mean')
        parser.add_argument('--std', default='', help='comma separated image std')
        parser.add_argument('--seed', default=42, type=int, help='random seed used e.g. for dataset subset selection')
        parser.add_argument('--preload', action='store_true', default=False, help='preload dataset in memory')
        parser.add_argument('--epoch-size', type=int, default=1000, help='epoch size')

        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--debug', action='store_true', default=False, help='debug mode: everything goes to TMP dir')
        parser.add_argument('--no-val', dest='val', action='store_false', help='do not run validation')

        # network saving and loading parameters
        parser.add_argument('--save-epoch-freq', default=10, type=int, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--load-path', type=str, help='path of the model to load')
        parser.add_argument('--epoch-count', default=0, type=int, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        # training parameters
        parser.add_argument('--nepoch', type=int, help='# of iter at starting learning rate')
        parser.add_argument('--nepoch-decay', default=0, type=int, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr', default=.00005, type=float, help='initial learning rate for adam')
        parser.add_argument('--lr-policy', default="linear", type=str, help='learning rate policy. [linear | step | plateau | cosine | exp | poly]')
        # validation parameters
        parser.add_argument('--val-freq', default=5, type=int, help='frequency of running validation step')

        # testing/prediction parameters
        parser.add_argument('--target-dir', default=None, help='target directory on which to make predictions (all images in folder)')
        parser.add_argument('--output-dir', default=None, help='output directory for saving predictions')
        parser.add_argument('--only-hm', action='store_true', default=False, help='whether to only save the heatmap (no gpkg)')
        parser.add_argument('--dtype', type=str, default="uint8", help='heatmap datatype [uint8 | float32 | uint16]')
        parser.add_argument('--ext', type=str, default=".tif", help='extension of the files to predict')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        if not opt.debug:
            # save to the disk
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt')
            with open(file_name + "_detailed.txt", 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
            with open(file_name + ".txt", 'wt') as opt_file:
                json.dump(opt.__dict__, opt_file, indent=2)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        if opt.load_from and os.path.exists(os.path.join(opt.load_from, 'opt.txt')):
            new_opt = {k: v for k, v in opt.__dict__.items() if v is not None}
            old_opt = copy.deepcopy(opt)
            optpath = os.path.join(opt.load_from, 'opt.txt')
            with open(optpath, 'r') as f:
                old_opt.__dict__ = json.load(f)
            # Overwrite old dict with new dict
            old_opt.__dict__.update(new_opt)
            opt = old_opt

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        # set bands
        if opt.pick_bands:
            bands = opt.pick_bands.split(",")
            opt.pick_bands = [int(b) for b in bands]
        if opt.mean:
            mean = opt.mean.split(",")
            opt.mean = [float(m) for m in mean]
        if opt.std:
            std = opt.std.split(",")
            opt.std = [float(s) for s in std]

        self.opt = opt
        return self.opt
