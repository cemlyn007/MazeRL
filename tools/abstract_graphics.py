import cv2
import numpy as np


class AbstractGraphics:
    def __init__(self, window_name: str) -> None:
        self.image = np.zeros([0, 0, 3], dtype=np.uint8)
        self.window_name = window_name

    def draw(self) -> None:
        raise NotImplementedError

    def show(self) -> None:
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)

    def save_image(self, filename: str) -> None:
        cv2.imwrite(filename, self.image)

    def _convert(self, pt: np.ndarray, magnification: int) -> np.ndarray:
        new_point = pt.copy()
        if new_point.ndim == 1:
            new_point[1] = 1.0 - new_point[1]
        else:
            new_point[:, 1] = 1.0 - new_point[:, 1]
        np.multiply(new_point, magnification, out=new_point)
        np.round(new_point, out=new_point)
        return new_point.astype(np.int32)