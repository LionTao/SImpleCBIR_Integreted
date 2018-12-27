import numpy as np
from modules.ObjectDetection.ObjDetectAPI import obj_dection


def render_color_bar_figure(color_vector: np.ndarray, path: str):
    import matplotlib.pyplot as plt

    plt.bar(x=range(48), height=color_vector.flatten().tolist(), color='bgr' * 16)
    plt.savefig(path)


def render_texture_bar_figure(texture_vector: np.ndarray, path: str):
    import matplotlib.pyplot as plt

    plt.bar(x=range(256), height=texture_vector.flatten().tolist())
    plt.savefig(path)


def ObjDetect(path: str):
    return obj_dection(path=path, dir="modules/ObjectDetection/")


if __name__ == '__main__':
    pass
