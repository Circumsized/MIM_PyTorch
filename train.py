import os
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mim import MIM
from dataset import MovingMNIST, RadarEcho, get_dataloader
from metrics import batch_mse, batch_psnr, batch_ssim, batch_mae, csi_score


def parse_args():
    parser = argparse.ArgumentParser(description='MIM Training')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'radar'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--total_length', type=int, default=20)
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, nargs='+',
                        default=[64, 64, 64, 64])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ss_start_epoch', type=int, default=0)
    parser.add_argument('--ss_stop_epoch', type=int, default=50)
    parser.add_argument('--ss_initial_prob', type=float, default=1.0)
    parser.add_argument('--ss_final_prob', type=float, default=0.0)
    parser.add_argument('--tln', action='store_true', default=True)
    parser.add_argument('--no_tln', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable automatic mixed precision training')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='enable torch.compile acceleration')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def get_scheduled_sampling_prob(epoch, args):
    if epoch < args.ss_start_epoch:
        return args.ss_initial_prob
    if epoch >= args.ss_stop_epoch:
        return args.ss_final_prob
    progress = (epoch - args.ss_start_epoch) / \
               max(1, args.ss_stop_epoch - args.ss_start_epoch)
    return args.ss_initial_prob - progress * \
        (args.ss_initial_prob - args.ss_final_prob)


def generate_ss_bool(batch_size, total_length, input_length,
                     height, width, prob, device):
    ss_length = total_length - input_length - 1
    ss_bool = (torch.rand(batch_size, ss_length, height, width,
                          device=device) < prob).float()
    return ss_bool


def train_one_epoch(model, dataloader, optimizer, criterion, epoch, args,
                    device, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None and device.type == 'cuda'

    ss_prob = get_scheduled_sampling_prob(epoch, args)

    for batch_idx, frames in enumerate(dataloader):
        frames = frames.to(device, non_blocking=True)
        batch_size = frames.shape[0]

        ss_bool = generate_ss_bool(
            batch_size, args.total_length, args.input_length,
            frames.shape[3], frames.shape[4], ss_prob, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                gen_imgs = model(frames, ss_bool)
                target = frames[:, 1:]
                loss = criterion(gen_imgs, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            gen_imgs = model(frames, ss_bool)
            target = frames[:, 1:]
            loss = criterion(gen_imgs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] '
                  f'Epoch {epoch}, Batch {batch_idx + 1}, '
                  f'Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}, '
                  f'SS Prob: {ss_prob:.3f}')

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, args, device):
    model.eval()
    metrics = {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'mae': 0.0}
    num_batches = 0

    for frames in dataloader:
        frames = frames.to(device, non_blocking=True)
        gen_imgs = model(frames, ss_bool=None)

        target = frames[:, args.input_length:]
        pred = gen_imgs[:, args.input_length - 1:]

        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        metrics['mse'] += batch_mse(pred, target)
        metrics['psnr'] += batch_psnr(pred, target)
        metrics['ssim'] += batch_ssim(pred, target)
        metrics['mae'] += batch_mae(pred, target)
        num_batches += 1

    for key in metrics:
        metrics[key] /= max(num_batches, 1)

    return metrics


def main():
    args = parse_args()

    if args.no_tln:
        args.tln = False

    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available()
                          else 'cpu')
    print(f'Using device: {device}')

    if args.dataset == 'mnist':
        img_channel = 1
        img_height = 64
        img_width = 64
        train_dataset = MovingMNIST(
            args.data_path, args.total_length, args.input_length,
            is_train=True)
        test_dataset = MovingMNIST(
            args.data_path, args.total_length, args.input_length,
            is_train=False)
    else:
        img_channel = 1
        img_height = 128
        img_width = 128
        train_dataset = RadarEcho(
            args.data_path, args.total_length, args.input_length,
            is_train=True)
        test_dataset = RadarEcho(
            args.data_path, args.total_length, args.input_length,
            is_train=False)

    train_loader = get_dataloader(
        train_dataset, args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    test_loader = get_dataloader(
        test_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    in_shape = [args.batch_size, img_channel, img_height, img_width]
    model = MIM(
        input_dims=img_channel,
        out_dims=img_channel,
        in_shape=in_shape,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        total_length=args.total_length,
        input_length=args.input_length,
        tln=args.tln
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {num_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    scaler = None
    if args.amp and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print('AMP enabled (float16)')

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device,
                                weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    if args.compile:
        print('Compiling model with torch.compile...')
        model = torch.compile(model)

    best_psnr = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, args, device,
            scaler=scaler)
        scheduler.step()

        print(f'Epoch {epoch} - Train Loss: {train_loss:.6f}')

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, test_loader, args, device)
            print(f'Epoch {epoch} - Eval: '
                  f'MSE={metrics["mse"]:.6f}, PSNR={metrics["psnr"]:.2f}, '
                  f'SSIM={metrics["ssim"]:.4f}, MAE={metrics["mae"]:.6f}')

            model_to_save = model._orig_mod \
                if isinstance(model, torch._dynamo.OptimizedModule) else model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'metrics': metrics,
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            save_path = os.path.join(
                args.save_dir, f'mim_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)

            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                best_path = os.path.join(args.save_dir, 'mim_best.pth')
                torch.save(checkpoint, best_path)
                print(f'New best PSNR: {best_psnr:.2f}')

    print('Training complete.')


if __name__ == '__main__':
    main()
