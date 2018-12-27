import tempfile


import cv2
import numpy as np

from modules.CBIRDataset.getDataset import GetDataSet
from modules.CNNCBIR.search_api import initCNNCache
from modules.ImageFeatureVector.HistogramVector.hisvec import get_vector
from modules.ImageFeatureExtract.imgfeature import feature_color, feature_shape, feature_texture
from modules.CNNCBIR.CreateMobileNet import extract_feat
from modules.CNNCBIR.search_api import search


def getDataset(des=''):
    # des = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    GetDataSet(des=des)
    print("DataSet should be ready at ", des + "/dataset")


# @Deprecated
def initCNNCBIR(dbpath):
    temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    initCNNCache(dataset_path=temp_dir, dbpath=temp_dir)
    print("Database Cached at", dbpath + "index.sqlite")


def make_image_cache(path, model):
    # vector

    img = cv2.imread(path)

    vector = np.array(get_vector(file=path))
    # colour
    color = np.array(feature_color(img_in=img))
    # shape
    shape = np.array(feature_shape(img_in=img))
    # texture
    texture = np.array(feature_texture(img_in=img))
    norm_feat = extract_feat(model=model, img_path=path)
    return [path, vector, color, shape, texture, norm_feat]


if __name__ == '__main__':
    getDataset()
