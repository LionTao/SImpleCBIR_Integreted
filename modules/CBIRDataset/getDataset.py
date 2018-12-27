def DownloadFile(file_url, file_path, md5check=False, md5=""):
    import requests
    import os

    # check dataset existence
    if os.path.isfile(file_path):
        print("[INFO]Dataset detected")
        if md5check:
            if md5 == "":
                raise Exception("md5 value not given")
            import hashlib
            existed_md5 = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            if existed_md5 == md5:
                print("[INFO]Dataset verified")
                return
        else:
            print("[CAUTION]Skipping md5 check")
        return

    # dataset not found
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


def ExtractZip(path, des="."):
    import zipfile

    print("\n[INFO]Start unzip")
    file_zip = zipfile.ZipFile(path, 'r')
    for file in file_zip.namelist():
        file_zip.extract(file, des + 'dataset/')
        print("\r Progress: {:.2f}%".format((file_zip.namelist().index(file) / len(file_zip.namelist())) * 100), end='')
    file_zip.close()
    print("[INFO]Unzipping Complete")


def GetDataSet(des=''):
    md5 = 'b2d5e7b89f3b1eed02f1e7cc61f929d5'
    import tempfile
    cdn_url = "http://media.liontao.xin/CorelDB.zip?token=8D-fPY7fZfvNQ_YlcCHphmf-beQ7s5-ahx1C_WJ4:B4fco3kqXcyC3Ast57tWaAxHCj4"

    file_name = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/SimpleCBIR_Corel10kDataset.zip'
    DownloadFile(file_url=cdn_url, file_path=file_name, md5check=True, md5=md5)
    ExtractZip(path=file_name, des=des)
    print("[INFO]Dataset is ready.")


if __name__ == '__main__':
    GetDataSet()
