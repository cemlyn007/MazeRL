import cv2
import numpy as np
import tools.abstract_graphics


class EpisodeRolloutTool(tools.abstract_graphics.AbstractGraphics):
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    def __init__(self, agentless_image: np.ndarray):
        super(EpisodeRolloutTool, self).__init__("Episode Rollout Policy")
        self._agentless_image = agentless_image
        if agentless_image.shape[0] != agentless_image.shape[1]:
            raise ValueError("Image is assumed to be square but this one isn't")
        # else...
        self.magnification = agentless_image.shape[0]
        self.orb_radius = int(0.02 * self.magnification)

    def set_states(self, states: np.ndarray) -> None:
        self.states = states

    def draw(self) -> None:
        self._draw_blank_environment()
        self._draw_greedy_path(self.states)

    def _draw_blank_environment(self) -> None:
        self.image = self._agentless_image.copy()

    def _draw_greedy_path(self, states: np.ndarray) -> None:
        thickness = int(0.004 * self.magnification)
        max_steps = len(states)
        for step_index in range(max_steps):
            state = self._convert(states[step_index])

            red = int(((max_steps - step_index) / max_steps) * 255)
            green = int((step_index / max_steps) * 255)
            color = (0, green, red)

            if step_index > 0:
                cv2.line(self.image, previous_state, state, color, thickness)
            previous_state = state

        cv2.circle(
            self.image, self._convert(states[0]), self.orb_radius, self.RED, cv2.FILLED
        )
        cv2.circle(
            self.image,
            self._convert(states[-1]),
            self.orb_radius,
            self.GREEN,
            cv2.FILLED,
        )

    def _convert(self, pt: np.ndarray) -> np.ndarray:
        return super()._convert(pt, self.magnification)


if __name__ == "__main__":
    agentless_image = np.zeros([500, 500, 3], dtype=np.uint8)
    tool = EpisodeRolloutTool(agentless_image)
    tool.set_states(np.array([[0.15, 0.15], [0.75, 0.85], [0.75, 0.15]]))
    tool.draw()
    tool.show()
    cv2.waitKey(0)

