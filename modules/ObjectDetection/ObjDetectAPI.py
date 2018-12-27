from modules.ObjectDetection.opencvdnn import do_object_detection as detect
import os


def obj_dection(path, network="MobileNet", modelDir="."):
    import requests
    if network == "MobileNet":
        prototxt = modelDir + "/MobileNetSSD_deploy.prototxt.txt"
        model = modelDir + "/MobileNetSSD_deploy.caffemodel"
    else:
        raise Exception("Network not provided")

    if not os.path.isfile(prototxt):
        file_url = "http://media.liontao.xin/MobileNetSSD_deploy.prototxt.txt?attname=&e=1545901370&token=8D-fPY7fZfvNQ_YlcCHphmf-beQ7s5-ahx1C_WJ4:eJEMiBlqucJesL3efchu7cHW7gU"
        file_path = prototxt

        from contextlib import closing
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"}
        with closing(requests.get(file_url, headers=headers, stream=True)) as response:
            chunk_size = 1024
            content_size = int(response.headers['content-length'])
            data_count = 0
            with open(file_path, "wb") as file:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    data_count = data_count + len(data)
                    now_jd = (data_count / content_size) * 100
                    print("\r Progress: {:.2f}% ({:d}/{:d}) - {}".format(now_jd, data_count, content_size, file_path),
                          end="")
            print("\r\n[INFO]Download Completed.")

    if not os.path.isfile(model):
        file_url = "http://media.liontao.xin/MobileNetSSD_deploy.caffemodel?attname=&e=1545901361&token=8D-fPY7fZfvNQ_YlcCHphmf-beQ7s5-ahx1C_WJ4:8DDL7DJyP0qnQt4b-tHBx9OsHS4"
        file_path = prototxt

        from contextlib import closing
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"}
        with closing(requests.get(file_url, headers=headers, stream=True)) as response:
            chunk_size = 1024
            content_size = int(response.headers['content-length'])
            data_count = 0
            with open(file_path, "wb") as file:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    data_count = data_count + len(data)
                    now_jd = (data_count / content_size) * 100
                    print("\r Progress: {:.2f}% ({:d}/{:d}) - {}".format(now_jd, data_count, content_size, file_path),
                          end="")
            print("\r\n[INFO]Download Completed.")

    if os.path.exists(path) and os.path.isfile(path) and os.path.isfile(prototxt) and os.path.isfile(model):
        image, category = detect(path, prototxt=prototxt, model=model)
        return image, category
    else:
        raise FileNotFoundError


if __name__ == '__main__':
    _, r = obj_dection(path="image.jpg")
    print(r)
