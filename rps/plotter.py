import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from collections import namedtuple
from dataclasses import dataclass, field


@dataclass
class EpisodeMetricPlot:
    ax:     Axes
    line:   Line2D
    data:   list[float] = field(default_factory=list)
    min:    float       = float("+inf")
    max:    float       = float("-inf")

    def clear(self):
        self.data.clear()
        self.min = float("+inf")
        self.max = float("-inf")


class AnimatedPlot:
    def __init__(self):
        self.x = []

        self.figure = None
        self.metrics: list[EpisodeMetricPlot] = []
        self.ax = None
        self.line = None

        self.smoothing_factor = 0.99

        self.animation = None

    def _update_plot(self, data: tuple):
        episode = data[0]
        self.x.append(episode)

        for metric, y in zip(self.metrics, data[1:]):
            smoothed_y = (1 - self.smoothing_factor) * y + self.smoothing_factor * (metric.data[-1] if metric.data else y)
            metric.data.append(smoothed_y)
            metric.min = metric.min if metric.min < smoothed_y else smoothed_y
            metric.max = metric.max if metric.max > smoothed_y else smoothed_y
            metric.line.set_data(self.x, metric.data)
            metric.ax.set_xlim(0, episode)
            metric.ax.set_ylim(metric.min, metric.max)

        # if len(self.x) > 100:
        #     self.x.clear()
        #     for metric in self.metrics:
        #         metric.clear()

        return self.line

    def _create_plot(self):
        self.figure, (ax_delta, ax_reward) = plt.subplots(nrows=1, ncols=2)
        
        line_delta, = ax_delta.plot(self.x, [])
        line_reward, = ax_reward.plot(self.x, [])

        ax_delta.set_title("Delta")
        ax_reward.set_title("Reward")

        self.metrics = [
            EpisodeMetricPlot(ax_delta, line_delta),
            EpisodeMetricPlot(ax_reward, line_reward),
        ]

    def setup(self, training_loop):
        self._create_plot()
        self.animation = FuncAnimation(self.figure, self._update_plot, frames=training_loop, save_count=100, interval=0)
        
    def start(self):
        plt.show()