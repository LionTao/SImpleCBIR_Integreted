import tempfile

import cv2
import numpy as np

from modules.CBIRDataset.getDataset import GetDataSet
from modules.CNNCBIR.search_api import initCNNCache
from modules.ImageFeatureVector.HistogramVector.hisvec import get_vector
from modules.ImageFeatureExtract.imgfeature import feature_color, feature_shape, feature_texture
from modules.CNNCBIR.CreateMobileNet import extract_feat
from modules.CNNCBIR.search_api import search
from modules.CNNCBIR.CreateMobileNet import initMobileNet
from main.models import cache
import os
from multiprocessing import Process


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
    res = {'path': path, 'vector': vector.tobytes(), 'color': color.tobytes(), 'shape': shape.tobytes(),
           'texture': texture.tobytes(), 'cnn': norm_feat.tobytes()}
    # return [path, vector, color, shape, texture, norm_feat]
    return res


def cacheAll():
    # from main.makecache import getDataset, make_image_cache

    model = initMobileNet()
    filepaths = list()
    for root, dirs, files in os.walk("dataset", topdown=False):
        for name in files:
            filepaths.append(os.path.join(os.path.abspath(root), name))
        # for name in dirs:
        #     print(os.path.join(root, name))

    for file in filepaths:
        pro_list = []
        if cache.objects.filter(path=file).count() > 0:
            print("existed")
            continue
        else:
            p = Process(target=cache_subfunc, args=(file, model,))
            p.start()
            pro_list.append(p)
        for p in pro_list:
            p.join()
        print("Cache Done")


def cache_subfunc(file, model):
    r = make_image_cache(file, model=model)
    try:
        cache.objects.create(**r)
    except Exception as e:
        print(e)
        cache.objects.filter(**r).update(**r)
    print(file)


if __name__ == '__main__':
    getDataset()
