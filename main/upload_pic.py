from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import json
import os
import tempfile
import cv2
from main.models import cache
from main.img_preprocess import preprocess_a, preprocess_b
from modules.CNNCBIR import search_api
import main.search_utils as search_utils
import numpy as np
from modules.ImageFeatureExtract.imgfeature import feature_color, feature_shape, feature_texture
import traceback
import io


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


@csrf_exempt
def upload_pic(request):
    return_message = dict()
    if request.method == 'POST':
        try:
            # Decode
            # ===============================
            encoded_json = request.body.decode('utf-8')
            json_body = json.loads(encoded_json)
            print(json_body)
            path = json_body['path']
            preprocessMethod = json_body['preprocessMethod']
            featureExtractionMethod = json_body['featureExtractionMethod']
            similarityCalculationMethod = json_body['similarityCalculationMethod']
            positionMethod = json_body['positionMethod']
            # ===============================
            print("[DEBUG] decode done")

            # Temp Dir
            # ===============================
            temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            # ===============================
            print("[DEBUG] tempdir done")

            # Compute
            # ===============================
            # Preprocess
            if preprocessMethod == '1-A':  # hist + 平滑
                usr_img = cv2.imread(path, 1)
                usr_img_pre = preprocess_a(usr_img)
                print("[DEBUG] preprocess func call done")
                usr_img_pre_dir = temp_dir + 'usr_img_pre.jpg'
                cv2.imwrite(usr_img_pre_dir, usr_img_pre)

                # following func accept image array not path
                search_utils.render_color_bar_figure(np.array(feature_color(img_in=usr_img_pre)),
                                                     temp_dir + 'usr_img_color.jpg')
                search_utils.render_texture_bar_figure(np.array(feature_texture(img_in=usr_img_pre)),
                                                       temp_dir + 'usr_img_texture.jpg')
                cv2.imwrite(temp_dir + 'usr_img_shape.jpg', np.array(feature_shape(img_in=usr_img_pre)))
            elif preprocessMethod == '1-B':  # hist + 锐化
                usr_img = cv2.imread(path, 1)
                usr_img_pre = preprocess_b(usr_img)
                print("[DEBUG] preprocess func call done")
                usr_img_pre_dir = temp_dir + 'usr_img_pre.jpg'
                cv2.imwrite(usr_img_pre_dir, usr_img_pre)

                # following func accept image array not path
                search_utils.render_color_bar_figure(np.array(feature_color(img_in=usr_img_pre)),
                                                     temp_dir + 'usr_img_color.jpg')
                search_utils.render_texture_bar_figure(np.array(feature_texture(img_in=usr_img_pre)),
                                                       temp_dir + 'usr_img_texture.jpg')
                cv2.imwrite(temp_dir + 'usr_img_shape.jpg', np.array(feature_shape(img_in=usr_img_pre)))
            else:
                raise Exception("Preprocess Method Unknown.")
            print("[DEBUG] process done")

            # Similarity
            res_num = 150
            # feats = cache.objects.values('cnn')
            if similarityCalculationMethod == '4-A':
                rawData = cache.objects.values_list('path', 'vector')
                imgNames, feats = list(), list()
                for element in rawData:
                    imgNames.append(element[0])
                    feats.append(convert_array(element[1]).reshape((20, 256, 1)))
                feats = np.array(feats)
                result, score = search_utils.vec_search(usr_img_pre_dir, feats, imgNames, res_num)
                # CNN part
                cnn_rawData = cache.objects.values_list('path', 'cnn')
                cnn_imgNames, cnn_feats = list(), list()
                for element in cnn_rawData:
                    cnn_imgNames.append(element[0])
                    cnn_feats.append(convert_array(element[1]))
                cnn_feats = np.array(cnn_feats)
                cnn_result, _ = search_api.search_with_cnnData(usr_img_pre_dir, cnn_feats, cnn_imgNames,
                                                               1)  # For getting the right category of the user picture
                category = os.path.split(cnn_result[0])[0].split("\\")[-1]
                score = [1 - i for i in score]  # 0 means same, 1 for different
            elif similarityCalculationMethod == '4-B':
                rawData = cache.objects.values_list('path', 'cnn')
                imgNames, feats = list(), list()
                for element in rawData:
                    imgNames.append(element[0])
                    feats.append(convert_array(element[1]))
                feats = np.array(feats)
                result, score = search_api.search_with_cnnData(usr_img_pre_dir, feats, imgNames, res_num)
                category = os.path.split(result[0])[0].split('\\')[-1]
            else:
                raise NotImplementedError("Similarity Method not Implemented")

            # Calculate precision and recall
            correct_number = 0
            for i in range(res_num):
                cate_temp = os.path.split(result[i])[0].split('\\')[-1]
                if cate_temp == category:
                    correct_number += 1
            precision = correct_number / res_num  # 查准率
            recall = correct_number / 100  # 查全率

            # Write result image to file
            for i in range(3):
                cv2.imwrite(temp_dir + "res{}.jpg".format(i + 1), cv2.imread(result[i]))
            print("[DEBUG] similarity done")

            # ObjectDetection
            if positionMethod == '3-A':
                objd_image, od_category = search_utils.ObjDetect(usr_img_pre_dir)
                cv2.imwrite(temp_dir + 'usr_objd.jpg', objd_image)
            else:
                raise Exception("Object Detection Method Unknown.")
            print("[DEBUG] User image OD done :", od_category)

            # Feature Extraction + OD
            if featureExtractionMethod == '2-A':  # FOO_BAR
                pass
            for i in range(3):
                db_dict = list(cache.objects.filter(path=result[i]).values())[0]
                # search_utils.render_color_bar_figure(db_dict['color'], temp_dir + 'color{}.jpg'.format(i + 1))
                # search_utils.render_texture_bar_figure(db_dict['texture'], temp_dir + 'texture{}.jpg'.format(i + 1))
                search_utils.render_color_bar_figure(feature_color(cv2.imread(db_dict['path'])),
                                                     temp_dir + 'color{}.jpg'.format(i + 1))
                search_utils.render_texture_bar_figure(feature_texture(cv2.imread(db_dict['path'])),
                                                       temp_dir + 'texture{}.jpg'.format(i + 1))
                cv2.imwrite(temp_dir + 'shape{}.jpg'.format(i + 1), feature_shape(cv2.imread(db_dict['path'])))
                od_temp, _ = search_utils.ObjDetect(result[i])
                cv2.imwrite(temp_dir + 'position{}.jpg'.format(i + 1), od_temp)

            print("[DEBUG] Feature Extraction + OD")

            # ===============================

            # Generate the json
            # ===============================
            # 0
            return_message['rawImage'] = usr_img_pre_dir  # the raw path
            # 0.5
            return_message['category'] = category  # 猜测的图片类别
            # 1
            return_message['recallRatio'] = recall  # "92.5%" 查全率
            # 2
            return_message['precision'] = precision  # "87.5%" 查准率
            # 3
            return_message['position'] = temp_dir + 'usr_objd.jpg'  # path of the target marked image
            # 4
            features = dict()
            features['color'] = temp_dir + 'usr_img_color.jpg'  # path of the generated image
            features['texture'] = temp_dir + 'usr_img_texture.jpg'  # path of the generated image
            features['shape'] = temp_dir + 'usr_img_shape.jpg'  # path of the generated image
            return_message['features'] = features
            features['position'] = temp_dir + 'usr_objd.jpg'  # path of the target marked image

            # 5
            results = dict()
            # 5.1
            one = dict()
            one['similarity'] = score[0].astype(float)
            one['path'] = temp_dir + 'res1.jpg'
            one_features = dict()
            one_features['color'] = temp_dir + 'color1.jpg'
            one_features['texture'] = temp_dir + 'texture1.jpg'
            one_features['shape'] = temp_dir + 'shape1.jpg'
            one_features['position'] = temp_dir + 'position1.jpg'
            one['features'] = one_features
            # 5.2
            two = dict()
            two['similarity'] = score[1].astype(float)
            two['path'] = temp_dir + 'res2.jpg'
            two_features = dict()
            two_features['color'] = temp_dir + 'color2.jpg'
            two_features['texture'] = temp_dir + 'texture2.jpg'
            two_features['shape'] = temp_dir + 'shape2.jpg'
            two_features['position'] = temp_dir + 'position2.jpg'
            two['features'] = two_features
            # 5.3
            three = dict()
            three['similarity'] = score[2].astype(float)
            three['path'] = temp_dir + 'res3.jpg'
            three_features = dict()
            three_features['color'] = temp_dir + 'color3.jpg'
            three_features['texture'] = temp_dir + 'texture3.jpg'
            three_features['shape'] = temp_dir + 'shape3.jpg'
            three_features['position'] = temp_dir + 'position3.jpg'
            three['features'] = three_features
            # 5
            results['one'] = one
            results['two'] = two
            results['three'] = three
            return_message['results'] = results
            # ===============================

            # Write the result to file
            # ===============================
            # # Try to make the temp folder
            # try:
            #     os.mkdir(temp_dir)
            # except Exception as x:
            #     print(str(x))

            # Try to write the json file
            try:
                busy = open(temp_dir + "result.json.busy", 'w')
                result_json = open(temp_dir + 'result.json', 'w')
                json.dump(return_message, result_json)
                result_json.close()
                busy.close()
                import time
                time.sleep(5)
                os.remove(temp_dir + 'result.json.busy')

            # Make sure everything is closed and deleted if anything bad occurs.
            except Exception as x:
                try:
                    busy.close()
                except Exception as e:
                    print(str(e))
                    print("Busy Doesn't Exist.")
                try:
                    os.remove(temp_dir + 'result.json.busy')
                except Exception as e:
                    print(str(e))
                    print("Busy Doesn't Exist.")
                try:
                    result_json.close()
                except Exception as e:
                    print(str(e))
                    print("Result Doesn't Exist.")
                try:
                    os.remove(temp_dir + 'result.json')
                except Exception as e:
                    print(str(e))
                    print("Result Doesn't Exist.")

                # Raise the exception to the upper level.
                traceback.print_exc()
                raise x
            # ===============================

        except Exception as e:
            traceback.print_exc()
            return HttpResponseBadRequest(str(e))
    else:
        return HttpResponseBadRequest('Not a POST request.')

    # Return http status 200
    return HttpResponse()
