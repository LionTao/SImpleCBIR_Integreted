import tempfile
import os
import cv2
import numpy as np

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from modules.CBIRDataset.getDataset import GetDataSet
from modules.CNNCBIR.search_api import initCNNCache
from modules.ImageFeatureVector.HistogramVector.hisvec import get_vector
from modules.ImageFeatureExtract.imgfeature import feature_color, feature_shape, feature_texture
from modules.CNNCBIR.CreateMobileNet import extract_feat
from modules.CNNCBIR.search_api import search
from modules.CNNCBIR.CreateMobileNet import initMobileNet
from main.models import cache
from queue import Queue
import tensorflow
import time
import io
import sqlite3

tf_config = tensorflow.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tensorflow.Session(config=tf_config)


def getDataset(des=''):
    # des = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    GetDataSet(des=des)
    print("DataSet should be ready at ", des + "/dataset")


# @Deprecated
def initCNNCBIR(dbpath):
    temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    initCNNCache(dataset_path=temp_dir, dbpath=temp_dir)
    print("Database Cached at", dbpath + "index.sqlite")


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def make_image_cache(path, model):
    # vector
    vector = np.array(get_vector(file=path))
    norm_feat = extract_feat(model=model, img_path=path)
    res = {'path': path, 'vector': adapt_array(vector), 'cnn': adapt_array(norm_feat)}
    return res


def cacheAll():
    # from main.makecache import getDataset, make_image_cache
    filepaths = list()
    for root, dirs, files in os.walk("dataset", topdown=False):
        for name in files:
            filepaths.append(os.path.join(os.path.abspath(root), name))
        # for name in dirs:
        #     print(os.path.join(root, name))

    # q = Queue()
    model = initMobileNet()
    i = 0
    for file in filepaths:
        i += 1
        t = time.time()
        # pro_list = []
        if cache.objects.filter(path=file).count() > 0:
            print("\r{:.2f}% existed".format(i / len(filepaths) * 100), end='')
            continue
        else:
            r = make_image_cache(file, model=model)
            try:
                cache.objects.create(**r)
                print("\r" + file + " Progress:{:.2f}%".format(i / len(filepaths) * 100) + " ETA:{:.1f}min".format(
                    (time.time() - t) * (len(filepaths) - i) / 60), end='')
            except Exception as e:
                print(e)
                cache.objects.filter(path=r["path"]).update(**r)
    print("\nCache Done")


if __name__ == '__main__':
    getDataset()
