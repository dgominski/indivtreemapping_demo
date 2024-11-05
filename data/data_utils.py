import numpy as np
import torch


def gkern(size=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel


def draw_gaussian_windowed(shape, centers, sigmas, amplitudes=None, pool="max"):
    """ for each center point in centers, draw a gaussian around that point with sigma dispersion and unit amplitude (if not provided) """
    WINDOW_SIZE = 149
    if not WINDOW_SIZE & 0x1:
        raise ValueError("only supports odd window size")

    #array = np.zeros([len(centers)] + [*shape])
    array = np.zeros(shape)
    if amplitudes is None:
        amplitudes = np.ones(len(centers))

    centers = centers.astype(int)
    for i, (x, y) in enumerate(centers):
        kernel = amplitudes[i] * gkern(WINDOW_SIZE, sigmas[i])
        tkernel = kernel
        demiwindow = WINDOW_SIZE // 2
        if x < -demiwindow or y < -demiwindow or x > shape[0] + demiwindow or y > shape[1] + demiwindow:
            continue
        rowstart = x - demiwindow
        rowend = x + 1 + demiwindow
        colstart = y - demiwindow
        colend = y + 1 + demiwindow
        if rowstart < 0:
            tkernel = tkernel[-rowstart:]
            rowstart = 0
        if rowend > shape[0]:
            tkernel = tkernel[:shape[0]-rowend]
            rowend = shape[0]
        if colstart < 0:
            tkernel = tkernel[:, -colstart:]
            colstart = 0
        if colend > shape[1]:
            tkernel = tkernel[:, :shape[1]-colend]
            colend = shape[1]

        if pool == "max":
            array[rowstart:rowend, colstart:colend] = np.maximum(array[rowstart:rowend, colstart:colend], tkernel)
        elif pool == "sum":
            array[rowstart:rowend, colstart:colend] += tkernel
        else:
            raise ValueError("pooling method not supported")

    # keep one value per channel
    # array = np.amax(array, axis=0)
    # scale to always have unit amplitude
    #array /= array.max()
    #array /= array.sum()
    return array


def draw_gaussian_broadcast(shape, centers, sigmas, amplitudes=None, pool="max", dtype=torch.float32):
    """ for each center point in centers, draw a gaussian around that point with sigma dispersion and unit amplitude """
    with torch.cuda.amp.autocast(dtype=dtype):
        device = sigmas.device
        xs = torch.linspace(0, shape[0] - 1, steps=shape[0], device=device)
        ys = torch.linspace(0, shape[0] - 1, steps=shape[0], device=device)
        x, y = torch.meshgrid(xs, ys, indexing='ij')
        x = torch.stack(centers.shape[0] * [x])
        y = torch.stack(centers.shape[0] * [y])
        centers = centers.to(dtype)
        sigmas = sigmas.to(dtype)

        with torch.no_grad():
            distances_x = torch.exp(-0.5 * (x - centers[:, 0][:, None, None]) ** 2)
            distances_y = torch.exp(-0.5 * (y - centers[:, 1][:, None, None]) ** 2)
            densities = distances_x * distances_y
        densities = torch.pow(densities, 1/(sigmas[:, None, None]**2))

        # densities = torch.exp(-0.5 * ((x - centers[:, 0][:, None, None]) / sigmas[:, None, None]) ** 2) * torch.exp(-0.5 * ((y - centers[:, 1][:, None, None]) / sigmas[:, None, None]) ** 2)

        if amplitudes is not None:
            densities = amplitudes[:, None, None].to(device) * densities

        if pool == "max":
            density = torch.amax(densities, dim=0)
        elif pool == "sum":
            density = torch.sum(densities, dim=0)
        else:
            raise ValueError("pooling operation not recognized, pick sum or max")
    return density


def draw_gaussian_optimized(shape, centers, sigmas, ratio=2):
    """ for each center point in centers, draw a gaussian around that point with sigma dispersion and unit amplitude """
    device = sigmas.device
    xs = torch.linspace(0, shape[0] - 1, steps=shape[0] // ratio).to(device)
    ys = torch.linspace(0, shape[0] - 1, steps=shape[0] // ratio).to(device)
    x, y = torch.meshgrid(xs, ys, indexing='ij')
    x = torch.stack(centers.shape[0] * [x])
    y = x.permute(0, 2, 1)

    x_dist = (x - centers[:, 0][:, None, None]) / sigmas[:, None, None]
    y_dist = (y - centers[:, 1][:, None, None]) / sigmas[:, None, None]
    densities = torch.exp(-0.5 * torch.amin(x_dist ** 2 + y_dist ** 2, dim=0))

    #densities = torch.nn.functional.interpolate(densities.unsqueeze(0).unsqueeze(0), scale_factor=ratio, mode="bicubic", align_corners=True)[0, 0]
    return densities


def draw_gaussian_batched(shape, centers, sigmas, amplitudes=None, pool="max", max_chunk_size=10):
    """ for each center point in centers, draw a gaussian around that point with sigma dispersion and unit amplitude
     batched version to avoid memory issues"""
    device = sigmas.device
    xs = torch.linspace(0, shape[0] - 1, steps=shape[0]).to(device)
    ys = torch.linspace(0, shape[1] - 1, steps=shape[1]).to(device)
    xgrid, ygrid = torch.meshgrid(xs, ys, indexing='ij')

    density = torch.zeros(shape[0], shape[1]).to(device)
    if amplitudes is not None:
        for (center_batch, sigma_batch, amplitude_batch) in zip(torch.split(centers, max_chunk_size), torch.split(sigmas, max_chunk_size), torch.split(amplitudes, max_chunk_size)):
            x = torch.stack(center_batch.shape[0] * [xgrid]).detach()
            y = torch.stack(center_batch.shape[0] * [ygrid]).detach()

            densities = torch.exp(
                -0.5 * ((x - center_batch[:, 0][:, None, None]) / sigma_batch[:, None, None]) ** 2) * torch.exp(
                -0.5 * ((y - center_batch[:, 1][:, None, None]) / sigma_batch[:, None, None]) ** 2)
            # print(densities.size())
            densities = amplitude_batch[:, None, None].to(device) * densities

            if pool == "max":
                density = torch.maximum(density, torch.amax(densities, dim=0))
            elif pool == "sum":
                density = density + torch.sum(densities, dim=0)
            else:
                raise ValueError("pooling operation not recognized, pick sum or max")
    else:
        for (center_batch, sigma_batch) in zip(torch.split(centers, max_chunk_size), torch.split(sigmas, max_chunk_size)):
            x = torch.stack(center_batch.shape[0] * [xgrid]).detach()
            y = torch.stack(center_batch.shape[0] * [ygrid]).detach()

            densities = torch.exp(-0.5 * ((x - center_batch[:, 0][:, None, None]) / sigma_batch[:, None, None]) ** 2) * torch.exp(-0.5 * ((y - center_batch[:, 1][:, None, None]) / sigma_batch[:, None, None]) ** 2)
            #print(densities.size())
            if amplitudes is not None:
                densities = amplitudes[:, None, None].to(device) * densities

            if pool == "max":
                density = torch.maximum(density, torch.amax(densities, dim=0))
            elif pool == "sum":
                density = density + torch.sum(densities, dim=0)
            else:
                raise ValueError("pooling operation not recognized, pick sum or max")

    return density
