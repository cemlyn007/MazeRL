import math
import time

import cv2
import numpy as np

import discrete_agent
import discrete_dqns.dqn
import environments.basic_environment
import tools.abstract_graphics

Point = tuple[float, float]


class ActionsVisualTool(tools.abstract_graphics.AbstractGraphics):
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)

    def __init__(self, magnification: int, n_cells: int, n_actions: int,
                 agent: discrete_agent.DiscreteAgent):
        super().__init__('Actions Qs', magnification, agent)
        self.n_cells = n_cells
        self.n_actions = n_actions
        self.agent = agent
        self.image = np.zeros([int(self.magnification),
                               int(self.magnification), 3],
                              dtype=np.uint8)

    def grids(self) -> list[tuple[Point, Point]]:
        grids = []
        for i in range(1, self.n_cells):
            pos = i / self.n_cells
            grids.extend((((0., pos), (1., pos)),
                          ((pos, 0.), (pos, 1.))))
        return grids

    def draw(self) -> None:
        self.image.fill(0)
        unit_pts = self._get_unit_square_pie()
        dt = 1. / self.n_cells / 2.
        for i in range(self.n_cells):
            x_mid = i / self.n_cells + dt
            for j in range(self.n_cells):
                y_mid = j / self.n_cells + dt
                state = np.array((x_mid, y_mid))
                q_values = self.agent.get_q_values(state)
                normalised = ((q_values - q_values.min())
                              / (q_values.max() - q_values.min()) * 255.).round().tolist()
                normalised = list(map(int, normalised))
                for k, pts in enumerate(unit_pts):
                    pts = np.array([[self.convert((x * dt + x_mid, y * dt + y_mid))
                                     for (x, y) in pts]],
                                   dtype=np.int32)
                    cv2.fillPoly(img=self.image, pts=pts,
                                 color=(255 - normalised[k], normalised[k], normalised[k]))

        for line in self.grids():
            cv2.line(self.image, self.convert(line[0]), self.convert(line[1]),
                     color=(255, 255, 255))

    def _get_unit_square_pie(self) -> list[tuple[Point, ...]]:
        polygons = []
        centre = (0., 0.)
        for k in range(self.n_actions):
            theta = math.pi * (2 * k - 1) / self.n_actions
            next_theta = math.pi * (2 * (k + 1) - 1) / self.n_actions
            corner_case = False
            for m in range(1, 10, 2):
                phi = m * math.pi / 4
                if theta % (2 * math.pi) < phi < next_theta % (2 * math.pi):
                    corner_case = True
                    break
            if corner_case:
                thetas = (theta, phi, next_theta)
            else:
                thetas = (theta, next_theta)

            cos_theta = np.cos(thetas)
            sin_theta = np.sin(thetas)
            abs_cos_theta = np.abs(cos_theta)
            abs_sin_theta = np.abs(sin_theta)
            inverse_cos_theta = np.where(abs_cos_theta > 0.0, 1 / abs_cos_theta, 1.0)
            inverse_sin_theta = np.where(abs_sin_theta > 0.0, 1 / abs_sin_theta, 1.0)
            rs = np.minimum(inverse_cos_theta, inverse_sin_theta)
            polygon = np.concatenate(
                (np.array([centre]), rs * np.array([cos_theta, sin_theta])), axis=0
            )

            polygons.append(tuple(polygon))
        return polygons


def _main():
    env = environments.basic_environment.BasicEnvironment(False, 500)
    agent = discrete_agent.DiscreteAgent(
        env,
        discrete_dqns.dqn.DiscreteDQN(discrete_dqns.dqn.helpers.Hyperparameters(), 4),
        4,
        0.01,
    )
    tool = ActionsVisualTool(500, 3, 4, agent)
    for i in range(100):
        tool.draw()
        tool.show()
        time.sleep(1.0)


if __name__ == '__main__':
    _main()
