import os
from torch.utils.tensorboard import SummaryWriter
from options.options import Options
from data import create_dataset
from models import create_model
import torch
import tqdm
from data.datahelpers import UnNormalize, AverageMeter


def val(valdataset, model, writer, epoch):
    model.eval()
    with torch.no_grad():
        print('Running validation...')
        val_loader = valdataset.get_loader()

        f1_averager = AverageMeter()
        loss_averager = AverageMeter()
        for i, data in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):  # inner loop within one epoch
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.forward()
            f1 = model.val()
            f1_averager.update(f1)
            loss_averager.update(model.loss_total)
        writer.add_scalar("val/{}_f1".format(valdataset.name), f1_averager.avg, global_step=epoch)
        writer.add_scalar("val/{}_loss".format(valdataset.name), loss_averager.avg, global_step=epoch)

    model.train()


def train(opt):
    traindataset = create_dataset(opt, name=opt.train_dataset, mode='train')  # create a dataset given opt.train_dataset_mode and other options
    valdataset = create_dataset(opt, name=opt.train_dataset, mode='val') if opt.val else None

    model = create_model(opt, mode="train")  # create a model given opt.model and other options
    total_iters = 0  # the total number of training iterations

    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "tbx")) if not opt.debug else SummaryWriter('/tmp')

    print(f'Train split size: {len(traindataset)}')
    print(f'Val split size: {len(valdataset)}') if opt.val else print("No validation")

    dataloader = traindataset.get_loader()
    unnorm = UnNormalize(mean=traindataset.normalizer.mean, std=traindataset.normalizer.std)

    for epoch in range(opt.epoch_count + 1, opt.nepoch + opt.nepoch_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        print('>> Epoch {}/{}: training...'.format(epoch, opt.nepoch + opt.nepoch_decay))
        for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):  # inner loop within one epoch
            total_iters += opt.batch_size

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                model.forward()
                model.compute_scalars()
            model.optimize_parameters(no_size=True)  # calculate loss functions, get gradients, update network weights
            if total_iters // opt.batch_size % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                writer.add_scalars("train/loss", losses, global_step=total_iters)
                model.compute_scalars()
                scalars = model.get_current_scalars()
                imgs = model.get_current_visuals_original()
                writer.add_scalars("train/scalars", scalars, global_step=total_iters)
                for n in imgs:
                    writer.add_images(f"train/{n}", imgs[n][:5], global_step=total_iters)
                writer.add_images(f"train/augmented_input", unnorm(model.input[:5, :3]), global_step=total_iters)

            if opt.lr_policy == "cyclical":
                model.update_learning_rate(verbose=False)  # update learning rates at the end of every batch for cyclical lr

        if epoch % opt.val_freq == 0 and opt.val:
            val(valdataset, model, writer, epoch)

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch {} / {}'.format(epoch, opt.nepoch + opt.nepoch_decay))

        model.update_learning_rate()  # update learning rates at the end of every epoch.

    print('saving the model at last epoch, iters %d' % (total_iters))
    model.save_networks('latest')


if __name__ == '__main__':
    opt = Options().parse()
    train(opt)





