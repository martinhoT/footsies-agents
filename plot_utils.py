import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.image import AxesImage
from typing import List


class Heatmap:
    def __init__(
        self,
        data: np.ndarray,
        xlabel: str = "x",
        ylabel: str = "y",
        xticks: List[str] = None,
        yticks: List[str] = None,
        title: str = "Heatmap",
    ):
        fig, ax = plt.subplots()
        im = ax.imshow(data)
        
        self.fig: Figure = fig
        self.ax: Axes = ax
        self.im: AxesImage = im

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if xticks:
            self.ax.set_xticks(np.arange(data.shape[1]), labels=xticks)
        if yticks:
            self.ax.set_yticks(np.arange(data.shape[0]), labels=yticks)

        self.text: List[List[Text]] = [
            [self.ax.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", color="black") for x in range(data.shape[1])]
            for y in range(data.shape[0])
        ]

        self.ax.set_title(title)

    def plot(self):
        self.fig.show()
    
    def update(self, data: np.ndarray):
        self.im.set_data(data)
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                self.text[y][x].set_text(f"{data[y, x]:.2f}")
        
        # TODO: what about draw_idle?
        self.fig.canvas.draw()
