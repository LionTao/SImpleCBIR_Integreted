import numpy as np
from modules.ObjectDetection.ObjDetectAPI import obj_dection


def render_color_bar_figure(color_vector: np.ndarray, path: str):
    import matplotlib.pyplot as plt

    plt.bar(x=range(48), height=color_vector.flatten().tolist(), color='bgr' * 16)
    plt.savefig(path)
    plt.close()


def render_texture_bar_figure(texture_vector: np.ndarray, path: str):
    import matplotlib.pyplot as plt

    plt.bar(x=range(256), height=texture_vector.flatten().tolist())
    plt.savefig(path)
    plt.close()


def ObjDetect(path: str):
    return obj_dection(path=path, dir="modules/ObjectDetection/")


def vec_search(imgpath: str, feats: np.ndarray, names: list, k=3):
    from modules.ImageFeatureVector.HistogramVector.hisvec import get_vector
    import cv2
    try:
        assert feats[0].shape == (20, 256, 1)
    except:
        raise Exception("assert feats shape failed", "expect (20, 256, 1)", " got ", feats[0].shape, " instead")
    v = get_vector(imgpath)
    assert v.shape == (20, 256, 1)
    nums = feats.shape[0]
    res = np.empty(nums)
    for i in range(nums):
        temp = np.empty(20)
        for j in range(20):
            temp[j] = (cv2.compareHist(v[j], feats[i][j], cv2.HISTCMP_BHATTACHARYYA))
        res[i] = np.linalg.norm(temp) / 20
    rank_ID = np.argsort(res)
    namelist = [names[index] for index in rank_ID[0:k]]
    sim = [res[index] for index in rank_ID[0:k]]  # similarities
    return namelist, sim
