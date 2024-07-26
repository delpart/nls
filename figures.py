from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

from model.lsystem import LSystem
from datasets import *

def dragon_gt():
        dragon = DragonCurve()
        gt = dragon.get_state(10)
        gt_noisy = dragon.get_state(10, noisy=True)

        figure_gt = dragon.get_svg_figure(gt)
        figure_gt_noisy = dragon.get_svg_figure(gt_noisy)
        
        with open(f'figures/dragon_noisy_gt.svg', 'w') as f:
            print(figure_gt_noisy, file=f)
        with open(f'figures/dragon_gt.svg', 'w') as f:
            print(figure_gt, file=f)
        
        t = dragon.get_state(10)
        t1 = dragon.get_state(11)

        figure_t = dragon.get_svg_figure(t)
        figure_t1 = dragon.get_svg_figure(t1)
        
        with open(f'figures/dragon_t.svg', 'w') as f:
            print(figure_t, file=f)
        with open(f'figures/dragon_t1.svg', 'w') as f:
            print(figure_t1, file=f)
        
        for tt, ttt in [('t', t), ('t1', t1)]:
            with open(f'figures/dragon_{tt}.txt', 'w') as f:
                s = []
                for c in ttt:
                    if c == 0:
                        s.append('F')
                    if c == 1:
                        s.append('G')
                    if c == 2:
                        s.append('+')
                    if c == 3:
                        s.append('-')
                s = ''.join(s)
                print(s, file=f)
        
        losses_normal = np.loadtxt('dragon2_loss.txt')
        losses_noisy = np.loadtxt('dragon2-noisy_loss.txt')
        lrs = []
        with open('dragon2-noisy_lr.txt', 'r') as f:
            for l in f.readlines():
                lrs.append(float(l[1:-2]))
        l_normal = np.zeros_like(losses_noisy)
        l_noisy = np.zeros_like(losses_noisy)
        l_normal[0] = losses_normal[0]
        l_noisy[0] = losses_noisy[0]
        for i in range(1, 1000):
            if i < len(losses_normal):
                l_normal[i] = l_normal[i-1]*0.8 + 0.2*losses_normal[i]
            l_noisy[i] = l_noisy[i-1]*0.8 + 0.2*losses_noisy[i]
        fig = plt.figure(figsize=(8, 4)) 
        plt.plot(lrs[:600], color='#ffc61e', label='Learning rate')
        plt.plot(l_normal[:600], color='#00cd6c', label='MSE unperturbed')
        plt.plot(l_noisy[:600], color='#af58ba', label='MSE noisy')
        plt.xlabel('epoch')
        plt.ylabel('MSE/Learning Rate')
        plt.legend()
        ax = plt.gca()
        ax.set_ylim([0, 0.5])
        plt.savefig('losses-lr.eps', format='eps')
        
if __name__ == '__main__':
    dragon_gt()

