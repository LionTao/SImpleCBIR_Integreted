def getDataset():
    import tempfile
    from modules.CBIRDataset.getDataset import GetDataSet
    temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    GetDataSet(des=temp_dir)
    print("DataSet should be ready at ", temp_dir + '/dataset')


def initCNNCBIR(dbpath):
    import tempfile
    from modules.CNNCBIR.search_api import initCNNCache
    temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    initCNNCache(dataset_path=temp_dir, dbpath=temp_dir)
    print("Database Cached at", dbpath + "index.sqlite")


def make_image_cache(path):
    from modules.ImageFeatureVector.HistogramVector.hisvec import get_vector
    from modules.ImageFeatureExtract.imgfeature import feature_color, feature_shape, feature_texture
    import cv2
    # vector

    img = cv2.imread(path)

    vector = get_vector(file=path)
    # colour
    color = feature_color(img_in=img)
    # shape
    shape = feature_shape(img_in=img)
    # texture
    texture = feature_texture(img_in=img)
    return [path, vector, color, shape, texture]


def render_color_bar_figure(color_vector: list, path: str):
    import matplotlib.pyplot as plt

    plt.bar(x=range(48), height=color_vector, color='bgr' * 16)
    plt.savefig(path)


def render_texture_bar_figure(texture_vector: list, path: str):
    import matplotlib.pyplot as plt

    plt.bar(x=range(256), height=texture_vector)
    plt.savefig(path)


if __name__ == '__main__':
    from modules.CNNCBIR.search_api import search

    # import tempfile
    #
    # temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
    # getDataset()
    # initCNNCBIR(dbpath=temp_dir)
    # res = search(
    #     imagepath="C:\\Users\\LionTao\\AppData\\Local\\Temp\\1\\SimpleCBIR_ResultTemp\\dataset\\art_1\\193000.jpg", k=3,
    #     dbpath="C:\\Users\\LionTao\\AppData\\Local\\Temp\\1\\SimpleCBIR_ResultTemp")
    # print(res)
    # res = make_image_cache(
    #     "C:\\Users\\LionTao\\AppData\\Local\\Temp\\1\\SimpleCBIR_ResultTemp\\dataset\\art_1\\193001.jpg")
    # print(res[3])
    # import cv2
    #
    # cv2.imshow("texture", res[3])
    # cv2.waitKey(0)

    # print(render_texture_bar_figure(res[3],"1.png"))
