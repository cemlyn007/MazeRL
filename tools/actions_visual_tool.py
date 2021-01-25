import cv2
import numpy as np
import torch

from .abstract_graphics import AbstractGraphics


class ActionsVisualTool(AbstractGraphics):
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)

    def __init__(self, magnification, agent, num_of_cells=10):
        super().__init__("Actions Qs", magnification, agent)
        self.num_of_cells = num_of_cells
        self.image = self.environment.image
        self.grid_centres = self.generate_grid_mid_points()
        self.x_axis = np.linspace(0, self.width * magnification,
                                  num=self.num_of_cells + 1,
                                  endpoint=True, dtype=np.int64)
        self.y_axis = np.linspace(0, self.height * magnification,
                                  num=self.num_of_cells + 1,
                                  endpoint=True, dtype=np.int64)
        self.triangles = self.generate_triangle_vertices()
        colors_shape = (self.num_of_cells, self.num_of_cells, 4, 3)
        self.bgr = np.zeros(colors_shape, dtype=np.uint8)
        self.thickness = max(int(0.002 * self.magnification), 1)
        self.states = self.get_state_space_tensor()

    def generate_grid_mid_points(self):
        x_shift = (self.width * .5) / self.num_of_cells
        xs = np.linspace(0.0 + x_shift, self.width - x_shift,
                         num=self.num_of_cells, endpoint=True,
                         dtype=np.float32)
        y_shift = (self.height * .5) / self.num_of_cells
        ys = np.linspace(0.0 + y_shift, self.height - y_shift,
                         num=self.num_of_cells, endpoint=True,
                         dtype=np.float32)
        ys = np.flip(ys)
        return np.dstack(np.meshgrid(xs, ys)) * self.magnification

    def generate_triangle_vertices(self):
        shape = (self.num_of_cells, self.num_of_cells, 4, 3, 2)
        triangles = np.zeros(shape)
        x = (self.width * .5) / self.num_of_cells
        y = (self.height * .5) / self.num_of_cells
        # Ordering: Right, Left, Up, Down (Deliberately to match action indexes)
        tri_orients = np.array([[[0, 0], [x, y], [x, -y]],
                                [[0, 0], [-x, y], [-x, -y]],
                                [[0, 0], [-x, -y], [x, -y]],
                                [[0, 0], [-x, y], [x, y]]]) * self.magnification
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    triangles[i, j, k] = (tri_orients[k] +
                                          self.grid_centres[i, j])
        return np.round(triangles).astype(np.int)

    def get_state_space_tensor(self):
        x_shift = (self.width * .5) / self.num_of_cells
        xs = np.linspace(0.0 + x_shift, self.width - x_shift,
                         num=self.num_of_cells, endpoint=True, dtype=np.float32)
        y_shift = (self.height * .5) / self.num_of_cells
        ys = np.linspace(0.0 + y_shift, self.height - y_shift,
                         num=self.num_of_cells, endpoint=True,
                         dtype=np.float32)
        states = torch.from_numpy(np.dstack(np.meshgrid(xs, ys)))
        return states

    def update_colors(self):
        was_training = False
        if self.dqn.q_network.training:
            self.dqn.q_network.eval()
            was_training = True
        with torch.no_grad():
            state = self.states.to(self.agent.dqn.device)
            predictions_tensor = self.dqn.q_network(state).cpu()
        if was_training:
            self.dqn.q_network.train()
        all_estimated_q_values = predictions_tensor.view(self.num_of_cells,
                                                         self.num_of_cells,
                                                         4).numpy()
        all_estimated_q_values /= all_estimated_q_values.sum(-1, keepdims=True)
        all_worst_to_best_triangles = np.argsort(all_estimated_q_values, -1)
        all_sorted_q_values = all_estimated_q_values.take(all_worst_to_best_triangles)

        def interpolate(x):
            return np.interp(x, x[[0, 3]], np.array((0., 255.), dtype=np.float32))

        reds = np.apply_along_axis(interpolate, -1, all_sorted_q_values)
        blues = reds[:, :, ::-1]
        greens = reds

        ordering = np.argsort(all_worst_to_best_triangles, -1)
        self.bgr[:, ..., 0] = blues.take(ordering)
        self.bgr[:, ..., 1] = greens.take(ordering)
        self.bgr[:, ..., 2] = reds.take(ordering)

    def draw_diagonals(self):
        color = (0, 0, 0)
        for x, y in zip(self.x_axis, self.y_axis):
            cv2.line(self.image, (x, 0), (0, y), color, self.thickness)
            cv2.line(self.image, (self.x_axis[-1] + x, 0),
                     (0, self.y_axis[-1] + y), color, self.thickness)
            cv2.line(self.image, (0, self.y_axis[-1] - y), (x, self.y_axis[-1]),
                     color, self.thickness)
            cv2.line(self.image, (x, 0), (self.x_axis[-1], self.y_axis[-1] - y),
                     color, self.thickness)

    def draw_borders(self):
        color = (255, 255, 255)
        for x in self.x_axis:
            cv2.line(self.image, (x, 0), (x, self.y_axis[-1]), color,
                     self.thickness)
        for y in self.y_axis:
            cv2.line(self.image, (0, y), (self.x_axis[-1], y), color,
                     self.thickness)

    def draw_triangles(self):
        self.update_colors()
        array_shape = self.triangles.shape
        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                for k in range(array_shape[2]):
                    cv2.fillConvexPoly(self.image,
                                       points=self.triangles[i, j, k],
                                       color=self.bgr[i, j, k].tolist()
                                       )

    def draw(self):
        self.image.fill(0)
        self.draw_triangles()
        self.draw_diagonals()
        self.draw_borders()
