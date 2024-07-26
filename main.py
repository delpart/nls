import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

from model.lsystem import LSystem
from datasets import *

def train(num_epochs=1000, use_ipex=False, eval_steps=10, eval_file='eval.svg', dataset='DragonCurve', rule_lengths=[1,3], load_model=False, noisy=False, p=0.01, batch_size=32):
    if use_ipex:
        import intel_extension_for_pytorch as ipex
        device = 'xpu'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data = getattr(importlib.import_module('datasets'), dataset)(noisy, p)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)

    lsystem = LSystem(rule_lengths=rule_lengths).to(device)
    if load_model:
        try:
            lsystem = torch.load(f'{eval_file}.pt')
        except:
            pass
    optim = torch.optim.SGD(lsystem.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=1e-7, max_lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.01, max_lr=0.1)
    if use_ipex:
        lsystem = lsystem.to(device)
        lsystem, optim = ipex.optimize(lsystem, optimizer=optim, dtype=torch.float32)

    losses = []
    lrs = []
    epochs = trange(num_epochs)
    for epoch in epochs:
        losses.append(0)
        batches = tqdm(enumerate(data_loader), leave=False)
        for j, batch in batches:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            y_hat = []
            for i in range(x.shape[-1]):
                t = lsystem(x[..., i:i+1])
                y_hat.append(t)
            y_hat = torch.concat(y_hat, dim=-1)

            loss = nn.functional.mse_loss(y_hat[:, :min(y.shape[-1], y_hat.shape[-1])], y[:, :min(y.shape[-1], y_hat.shape[-1])])
            loss.backward()
            if j+1 == len(data_loader):
                optim.step()
                optim.zero_grad()

            losses[-1] = (losses[-1]*j + loss.item())/(j+1)

            epochs.set_description(f'Loss [log10]: {np.log10(losses[-1])} - LR: {scheduler.get_last_lr()}')
        if epoch%100==0:
            torch.save(lsystem, f'{eval_file}.pt')
        lrs.append(scheduler.get_last_lr())
        scheduler.step()
    torch.save(lsystem, f'{eval_file}.pt')

    with torch.no_grad():
        gt = data.get_state(eval_steps)

        state = torch.zeros(1).to(device)
        state = state[None]

        for _ in trange(eval_steps):
            tmp = []
            for i in trange(state.shape[-1], leave=False):
                t = lsystem(state[..., i:i+1])
                tmp.append(t)
            state = torch.concat(tmp, dim=-1)
        state = torch.clip(state, 0, 3).long().squeeze(0).float()
        l = min(gt.shape[0], state.shape[0])
        print(nn.functional.mse_loss(torch.tensor(gt[:l]).float().to(device), state[:l]))
        print(f'Num atoms: {l}')
        state = state.cpu().numpy()
        # plt.scatter(range(l), gt[:l], label='gt', alpha=0.5)
        # plt.scatter(range(l), state[:l], label='model', alpha=0.5)
        # plt.legend()
        # plt.show()
        
        figure_pred = data.get_svg_figure(state)
        figure_gt = data.get_svg_figure(gt)
        
        with open(f'{eval_file}_prediction.svg', 'w') as f:
            print(figure_pred, file=f)
        with open(f'{eval_file}_gt.svg', 'w') as f:
            print(figure_gt, file=f)
        with open(f'{eval_file}_loss.txt', 'w') as f:
            print('\n'.join(map(str, losses)), file=f)
        with open(f'{eval_file}_lr.txt', 'w') as f:
            print('\n'.join(map(str, lrs)), file=f)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--use-ipex', action='store_true')
    parser.add_argument('--eval-steps', type=int, default=10)
    parser.add_argument('--eval-file', type=Path, default='dragoncurve')
    parser.add_argument('--dataset', type=str, default='DragonCurve')
    parser.add_argument('--rule-lengths', nargs='+', type=int, default=[1, 3])
    parser.add_argument('-c', action='store_true')
    parser.add_argument('--noisy', action='store_true')
    parser.add_argument('-p', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    train(args.num_epochs, args.use_ipex, args.eval_steps, args.eval_file, args.dataset, args.rule_lengths, args.c, args.noisy, args.p, args.batch_size)

