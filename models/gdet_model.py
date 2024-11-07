import numpy as np
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models import create_model
import segmentation_models_pytorch as smp
import tqdm
from layers.loss import AdaptiveHeatmapLossFromCenters
from util.evaluate import evaluate
from skimage.feature import peak_local_max


class GDetModel(BaseModel, torch.nn.Module):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        torch.nn.Module.__init__(self)
        print(f"## GDet model")
        self.loss_names = ["scale_reg", "heatmap", "total"]
        self.scalar_names = ["f1", "size_mean", "size_std"]
        self.model_names = ['unet_encoder', 'unet_decoder', 'heatmap_head', 'sigma_head']
        self.visual_names = ["visual_heatmap", "visual_sigmas", "visual_input", "scaled_target"]

        unet = smp.Unet("resnet50", in_channels=3, encoder_weights='imagenet', classes=1)
        if opt.nbands != 3:
            unet.encoder.conv1 = torch.nn.Conv2d(opt.nbands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if opt.load_from is None:
            bigearthnet_ckpt_url = "https://sid.erda.dk/share_redirect/gurAPaeeZY/latest_net_feature_extractor.pth"
            bigearthnet_ckpt = torch.hub.load_state_dict_from_url(bigearthnet_ckpt_url)
            bigearthnet_ckpt = {x.replace("module.0", "conv1"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.1", "bn1"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.4", "layer1"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.5", "layer2"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.6", "layer3"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.7", "layer4"): y for x, y in bigearthnet_ckpt.items()}
            unet.encoder.load_state_dict(bigearthnet_ckpt)
        self.unet_encoder = unet.encoder.to(self.device)
        self.unet_decoder = unet.decoder.to(self.device)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=(3, 3), padding=1, bias=True)).to(self.device)
        self.sigma_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=(3, 3), padding=1, bias=True)).to(self.device)

        self.optimizers["main"] = torch.optim.Adam(list(self.unet_encoder.parameters())
                                                   + list(self.unet_decoder.parameters())
                                                , lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        self.optimizers["heads"] = torch.optim.Adam(list(self.heatmap_head.parameters()) + list(self.sigma_head.parameters())
                                                   , lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)

        self.heatmap_criterion = AdaptiveHeatmapLossFromCenters(reference_ground_resolution=opt.gnd, pr_min=3*opt.sigma/4)

        self.loss_scaler = torch.cuda.amp.GradScaler()

    def set_input(self, inputdict):
        self.input = inputdict.get('input', None).to(self.device).float()
        self.visual_input = inputdict.get('rawinput', None).float()
        self.ground_resolution = inputdict.get('resolution', None)
        self.valid = inputdict.get('valid', None).to(torch.float16).to(self.device)
        self.weightmap = inputdict.get('weightmap', None).to(torch.float16).to(self.device)
        tcs = inputdict.get('centers', None)
        self.target_centers = [np.array(c) if c[0] is not None else np.empty((0, 2)) for c in tcs]

    def forward(self):
        # print(self.input.shape)
        deepfeats = self.unet_encoder(self.input)
        feats = self.unet_decoder(*deepfeats)
        self.heatmap = self.heatmap_head(feats)
        self.visual_heatmap = torch.clamp(self.heatmap, min=0, max=1)
        self.sigmas = self.sigma_head(feats)
        self.visual_sigmas = (self.sigmas.detach() - torch.amin(self.sigmas, dim=[2, 3], keepdim=True)) / (torch.amax(self.sigmas, dim=[2, 3], keepdim=True) - torch.amin(self.sigmas, dim=[2, 3], keepdim=True))
        return

    def __call__(self, image):
        self.input = image.to(self.device)
        self.forward()
        return self.heatmap

    def compute_loss(self):
        """Calculate loss"""
        self.loss_scale_reg, self.loss_heatmap, scaled_target = self.heatmap_criterion(self.heatmap, self.sigmas, self.weightmap, self.target_centers, self.ground_resolution, self.valid)
        self.scaled_target = torch.clamp(scaled_target, min=0, max=1)
        self.loss_total = self.loss_heatmap + self.opt.delta*self.loss_scale_reg
        if self.loss_total.isnan():
            raise ValueError("loss went crazy")

    def optimize_parameters(self, no_size=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.compute_loss()
        for name in self.optimizers:
            self.optimizers[name].zero_grad()
        # self.loss_total.backward()
        self.loss_scaler.scale(self.loss_total).backward()
        for name in self.optimizers:
            # self.optimizers[name].step()
            self.loss_scaler.step(self.optimizers[name])
        self.loss_scaler.update()
        return

    def compute_scalars(self):
        with torch.no_grad():
            preds = [p[0].cpu().numpy().astype(float) for p in self.heatmap]
            results = evaluate(self.target_centers, preds, min_distance=self.opt.nms_kernel_size, threshold_rel=None, threshold_abs=self.opt.threshold, max_distance=self.opt.gamma)
            self.f1 = np.nanmean(results["fscore"])
            self.size_mean = torch.mean(self.sigmas, dim=0).mean()
            self.size_std = torch.std(self.sigmas, dim=0).mean()
        return self.f1

    def val(self):
        with torch.no_grad():
            self.compute_loss()
            f1 = self.compute_scalars()
        return f1

    def decode(self):
        dets = []
        for i in range(self.heatmap.shape[0]):
            pred = self.heatmap[i, 0].cpu().numpy()
            pred_indices = peak_local_max(pred, min_distance=self.opt.nms_kernel_size, threshold_abs=self.opt.threshold)
            dets.append(pred_indices)
        return dets


if __name__ == '__main__':
    from options.options import Options
    from data.datahelpers import pil_loader, custom_collate
    from data import create_dataset
    from torch.utils.tensorboard import SummaryWriter
    from util.evaluate import DictAverager
    import rasterio
    np.random.seed(42)

    opt = Options()
    opt = opt.parse()
    model = create_model(opt, mode='test')

    traindataset = create_dataset(opt, name=opt.train_dataset, mode='val')
    loader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, collate_fn=custom_collate)
    with torch.no_grad():
        for nms_k_size in [1, 3, 5]:
            for thresh in [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]:
                model.decoder.nms_kernel_size = nms_k_size
                model.decoder.threshold = thresh
                averager = DictAverager()
                for i, data in tqdm.tqdm(enumerate(loader), total=len(loader)):  # inner loop within one epoch
                    model.set_input(data)
                    model.forward()
                    _, resdet = model.compute_scalars()
                    averager.update(resdet)
                print(nms_k_size, thresh, averager.get_avg())