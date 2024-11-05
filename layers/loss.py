import torch
import torch.nn as nn
from data.data_utils import draw_gaussian_broadcast

# --------------------------------------
# Loss/Error layers
# --------------------------------------


class AdaptiveHeatmapLossFromCenters(nn.Module):
    def __init__(self, reference_ground_resolution, pr_min=.2):
        super(AdaptiveHeatmapLossFromCenters, self).__init__()
        self.reference_ground_resolution = reference_ground_resolution  # image-level scale factors (not to be confused with scale predictions) computed as a ratio to this
        self.pr_min = pr_min  # in meters

    def forward(self, pred_hm, pred_sm, centers, ground_resolution, mask=None):
        scale_reg_losses = 0
        heatmaps_losses = 0
        scaled_gts = []
        for i in range(pred_hm.shape[0]):
            # scale loss
            scale_pred = pred_sm[i, 0]
            scale_loss = scale_pred ** 2
            scale_reg_losses += scale_loss.mean(dim=1).mean(dim=0)

            # heatmap loss
            if centers[i].size == 0:
                scaled_gt = torch.zeros_like(pred_hm[i, 0])
            else:
                gt_centers = torch.clamp(torch.Tensor(centers[i]).long(), min=0, max=pred_hm.shape[2]-1).to(pred_hm.device)
                gr = ground_resolution[i].to(pred_hm.device)
                predicted_sizes = (self.pr_min / gr) + nn.functional.relu(pred_sm[i, 0, gt_centers[:, 0], gt_centers[:, 1]]) * self.reference_ground_resolution / gr
                scaled_gt = draw_gaussian_broadcast((pred_hm.shape[2], pred_hm.shape[3]), gt_centers, sigmas=predicted_sizes)

            if mask is not None:
                m = mask[i, 0]
            else:
                m = torch.ones_like(pred_hm[i, 0])
            heatmaps_loss = (pred_hm[i, 0] - scaled_gt)**2 * m

            heatmaps_losses += heatmaps_loss.mean(dim=1).mean(dim=0)
            scaled_gts.append(scaled_gt)
        return scale_reg_losses / pred_hm.shape[0], heatmaps_losses / pred_hm.shape[0], torch.stack(scaled_gts).unsqueeze(1)

