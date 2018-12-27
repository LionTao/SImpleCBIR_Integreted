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


def vec_search(imgpath: str, feats: np.ndarray, names: list, k=3):
    # print(feats.shape);
    # exit()
    from modules.ImageFeatureVector.HistogramVector.hisvec import get_vector
    import cv2
    v = get_vector(imgpath)
    nums = feats.shape[0]
    res = np.empty(nums)
    for i in range(nums):
        temp = np.empty(20)
        for j in range(20):
            temp[i] = (cv2.compareHist(v[j], feats[i][j], cv2.HISTCMP_BHATTACHARYYA))
        res[i] = np.linalg.norm(temp) / 20
    print(res.argsort(res))
