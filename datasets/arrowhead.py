import torch
from torch.utils.data import Dataset
import numpy as np
import svg


class ArrowheadCurve(Dataset):
    def __init__(self, noisy=False, p=0.01):
        super().__init__()
        self.X = []
        self.noisy = noisy
        self.p = p
        state = np.zeros((1,))
        for _ in range(5):
            self.X.append(state)
            tmp = []
            for i in range(state.shape[0]):
                if state[i] == 0:
                    t = np.zeros(5)
                    t[0] = 1
                    t[1] = 3
                    t[2] = 0
                    t[3] = 3
                    t[4] = 1
                elif state[i] == 1:
                    t = np.zeros(5)
                    t[0] = 0
                    t[1] = 2
                    t[2] = 1
                    t[3] = 2
                    t[4] = 0
                elif state[i] == 2:
                    t = np.ones(1)*2
                elif state[i] == 3:
                    t = np.ones(1)*3
                if noisy and np.random.rand() < p:
                    t = np.random.randint(0, 4, t.shape)
                tmp.append(t)
            state = np.concatenate(tmp, axis=0)

    def __len__(self):
        return len(self.X) - 1

    def __getitem__(self, item):
        x, y = torch.from_numpy(self.X[item]).float(), torch.from_numpy(self.X[item+1]).float()
        return x, y

    def get_svg_figure(self, string):
        points = [0, 0]
        direction = 0
        
        for c in string:
            if np.rint(c) == 0 or np.rint(c) == 1:
                points.append(points[-2] + np.sin(direction*np.pi/3)*10)
                points.append(points[-2] + np.cos(direction*np.pi/3)*10)
            elif np.rint(c) == 2:
                direction = (direction+1)%6
            elif np.rint(c) == 3:
                direction = (direction-1)%6

        min_x = min(points[::2])
        min_y = min(points[1::2])

        for i in range(len(points)):
            if i%2 == 0:
                points[i] -= min_x
            else:
                points[i] -= min_y

        max_x = max(points[::2])
        max_y = max(points[1::2])

        line = svg.Polyline(points=points, fill='transparent', stroke_width=1, stroke='black')
        figure = svg.SVG(width=max_x, height=max_y, elements=[line])
        return figure

    def get_state(self, steps=10, noisy=False, p=0.01):
        state = np.zeros((1,))
        for _ in range(steps):
            tmp = []
            for i in range(state.shape[0]):
                if state[i] == 0:
                    t = np.zeros(5)
                    t[0] = 1
                    t[1] = 3
                    t[2] = 0
                    t[3] = 3
                    t[4] = 1
                elif state[i] == 1:
                    t = np.zeros(5)
                    t[0] = 0
                    t[1] = 2
                    t[2] = 1
                    t[3] = 2
                    t[4] = 0
                elif state[i] == 2:
                    t = np.ones(1)*2
                elif state[i] == 3:
                    t = np.ones(1)*3
                if noisy and np.random.rand() < p:
                    t = np.random.randint(0, 4, t.shape)
                tmp.append(t)
            state = np.concatenate(tmp, axis=0)
        return state
