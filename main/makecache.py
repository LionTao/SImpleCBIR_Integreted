import tempfile
import cv2
import numpy as np

from modules.CBIRDataset.getDataset import GetDataSet
from modules.CNNCBIR.search_api import initCNNCache
from modules.ImageFeatureVector.HistogramVector.hisvec import get_vector
from modules.ImageFeatureExtract.imgfeature import feature_color, feature_shape, feature_texture
from modules.CNNCBIR.CreateMobileNet import extract_feat
from modules.CNNCBIR.search_api import search


def getDataset():
    temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    GetDataSet(des=temp_dir)
    print("DataSet should be ready at ", temp_dir + '/dataset')


def initCNNCBIR(dbpath):
    temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    initCNNCache(dataset_path=temp_dir, dbpath=temp_dir)
    print("Database Cached at", dbpath + "index.sqlite")


def make_image_cache(path,model):
    # vector

    img = cv2.imread(path)

    vector = np.array(get_vector(file=path))
    # colour
    color = np.array(feature_color(img_in=img))
    # shape
    shape = np.array(feature_shape(img_in=img))
    # texture
    texture = np.array(feature_texture(img_in=img))
    norm_feat = extract_feat(model=model,img_path=path)
    return [path, vector, color, shape, texture, norm_feat]


def render_color_bar_figure(color_vector: np.ndarray, path: str):
    import matplotlib.pyplot as plt

    plt.bar(x=range(48), height=color_vector.flatten().tolist(), color='bgr' * 16)
    plt.savefig(path)


def render_texture_bar_figure(texture_vector: np.ndarray, path: str):
    import matplotlib.pyplot as plt

    plt.bar(x=range(256), height=texture_vector.flatten().tolist())
    plt.savefig(path)


if __name__ == '__main__':
    # temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    # getDataset()
    # initCNNCBIR(dbpath=temp_dir)
    # res = search(
    #     imagepath="C:\\Users\\LionTao\\AppData\\Local\\Temp\\1\\SimpleCBIR_ResultTemp\\dataset\\art_1\\193000.jpg", k=3,
    #     dbpath="C:\\Users\\LionTao\\AppData\\Local\\Temp\\1\\SimpleCBIR_ResultTemp")
    # print(res)
    from modules.CNNCBIR.CreateMobileNet import initMobileNet
    res = make_image_cache(
        "C:\\Users\\LionTao\\AppData\\Local\\Temp\\1\\SimpleCBIR_ResultTemp\\dataset\\art_1\\193002.jpg",model=initMobileNet())
    print(res[4])

    # cv2.imshow("texture", res[3])
    # cv2.waitKey(0)

    print(render_color_bar_figure(res[2], "1.png"))
