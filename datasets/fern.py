import torch
from torch.utils.data import Dataset
import numpy as np
import svg


class Fern(Dataset):
    def __init__(self, noisy=False, p=0.01):
        super().__init__()
        self.X = []
        state = np.zeros((1,))
        for _ in range(7):
            self.X.append(state)
            tmp = []
            for i in range(state.shape[0]):
                if state[i] == 0:
                    t = np.zeros(18)
                    t[0] = 1
                    t[1] = 2
                    t[2] = 4
                    t[3] = 4
                    t[4] = 0
                    t[5] = 5
                    t[6] = 3
                    t[7] = 0
                    t[8] = 5
                    t[9] = 4
                    t[10] = 1
                    t[11] = 4
                    t[12] = 3
                    t[13] = 1
                    t[14] = 0
                    t[15] = 5
                    t[16] = 2
                    t[17] = 0
                elif state[i] == 1:
                    t = np.zeros(2)
                    t[0] = 1
                    t[1] = 1
                else:
                    t = np.ones(1)*state[i]
                if noisy and np.random.rand() < p:
                    t = np.random.randint(0, 6, t.shape)
                tmp.append(t)
            state = np.concatenate(tmp, axis=0)

    def __len__(self):
        return len(self.X) - 1

    def __getitem__(self, item):
        return torch.from_numpy(self.X[item]).float(), torch.from_numpy(self.X[item+1]).float()

    def get_svg_figure(self, string):
        start_points = []
        end_points = []
        direction = 0
        position = 0, 0
        
        stack = []
        
        for c in string:
            if np.rint(c) == 1:
                start_points.append(position[0])
                start_points.append(position[1])
                position = position[0] + np.sin(direction*np.pi/6)*10, position[1] + np.cos(direction*np.pi/6)*10
                end_points.append(position[0])
                end_points.append(position[1])
            elif np.rint(c) == 2:
                direction = (direction + 1)%12
            elif np.rint(c) == 3:
                direction = (direction - 1)%12
            elif np.rint(c) == 4:
                stack.append((position, direction))
            elif np.rint(c) == 5:
                position, direction = stack[-1]
                stack = stack[:-1]
        
        if len(start_points) <= 0:
            return svg.SVG()

        min_x =  min(min(start_points[::2]),min(end_points[::2]))
        min_y =  min(min(start_points[1::2]),min(end_points[1::2]))
        
        for i in range(len(start_points)):
            if i%2 == 0:
                start_points[i] -= min_x
                end_points[i] -= min_x
            else:
                start_points[i] -= min_y
                end_points[i] -= min_y
        
        max_x =  max(max(start_points[::2]),max(end_points[::2]))
        max_y =  max(max(start_points[1::2]),max(end_points[1::2]))

        lines = []
        
        for i in range(len(start_points)//2):
            line = svg.Line(x1=start_points[2*i], y1=start_points[2*i+1],
                            x2=end_points[2*i], y2=end_points[2*i+1],
                            stroke_width=1, stroke='black'
                            )
            lines.append(line)
        figure = svg.SVG(width=max_x, height=max_y, elements=lines)
        return figure

    def get_state(self, steps=10, noisy=False, p=0.01):
        state = np.zeros((1,))
        for _ in range(steps):
            tmp = []
            for i in range(state.shape[0]):
                if state[i] == 0:
                    t = np.zeros(18)
                    t[0] = 1
                    t[1] = 2
                    t[2] = 4
                    t[3] = 4
                    t[4] = 0
                    t[5] = 5
                    t[6] = 3
                    t[7] = 0
                    t[8] = 5
                    t[9] = 3
                    t[10] = 1
                    t[11] = 4
                    t[12] = 3
                    t[13] = 1
                    t[14] = 0
                    t[15] = 5
                    t[16] = 2
                    t[17] = 0
                elif state[i] == 1:
                    t = np.zeros(2)
                    t[0] = 1
                    t[1] = 1
                else:
                    t = np.ones(1)*state[i]
                if noisy and np.random.rand() < p:
                    t = np.random.randint(0, 6, t.shape)
                tmp.append(t)
            state = np.concatenate(tmp, axis=0)
        return state
