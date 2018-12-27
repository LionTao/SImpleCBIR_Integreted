from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import json
import os
import tempfile
import cv2
from main.models import cache
from main.img_preprocess import preprocess
from modules.CNNCBIR import search_api
import main.search_utils as search_utils
import numpy as np
from modules.ImageFeatureExtract.imgfeature import feature_color, feature_shape, feature_texture


@csrf_exempt
def upload_pic(request):
    return_message = dict()
    if request.method == 'POST':
        try:
            # Decode
            # ===============================
            encoded_json = request.body.decode('utf-8')
            json_body = json.loads(encoded_json)
            path = json_body['path']
            preprocessMethod = json_body['preprocessMethod']
            featureExtractionMethod = json_body['featureExtractionMethod']
            similarityCalculationMethod = json_body['similarityCalculationMethod']
            positionMethod = json_body['positionMethod']
            # ===============================

            # Temp Dir
            # ===============================
            temp_dir = tempfile.gettempdir() + '/SimpleCBIR_ResultTemp/'
            # ===============================

            # Compute
            # ===============================
            # Preprocess
            if preprocessMethod == 'hist_canny':
                usr_img = cv2.imread(path, 1)
                usr_img_pre = preprocess(usr_img)
                usr_img_pre_dir = temp_dir + 'usr_img_pre.jpg'
                cv2.imwrite(usr_img_pre_dir, usr_img_pre)

                search_utils.render_color_bar_figure(np.array(feature_color(img_in=usr_img_pre_dir)),
                                                     temp_dir + 'usr_img_color.jpg')
                search_utils.render_texture_bar_figure(np.array(feature_texture(img_in=usr_img_pre_dir)),
                                                       temp_dir + 'usr_img_texture.jpg')
                cv2.imwrite(temp_dir + 'usr_img_shape.jpg', np.array(feature_shape(img_in=usr_img_pre_dir)))
            else:
                raise Exception("Preprocess Method Unknown.")

            # Similarity
            res_num = 3
            imgNames = cache.path.objects.all()
            feats = cache.cnn.objects.all()
            if similarityCalculationMethod == 'vector':
                result, score = search_utils.vec_search(usr_img_pre_dir, feats, imgNames, res_num)
            elif similarityCalculationMethod == 'cnn':
                result, score = search_api.search_with_cnnData(usr_img_pre_dir, feats, imgNames, res_num)

            for i in range(res_num):
                cv2.imwrite(temp_dir + "res{}.jpg".format(i + 1), result[i])

            # ObjectDetection
            if positionMethod == 'od':
                objd_image, catagory = search_utils.ObjDetect(usr_img_pre_dir)
                cv2.imwrite(temp_dir + 'usr_objd.jpg', objd_image)
            else:
                raise Exception("Object Detection Method Unknown.")

            # Feature Extraction + OD
            if featureExtractionMethod == 'FOO_BAR':
                pass
            for i in range(res_num):
                db_dict = list(cache.object.filter(path=result[i]).values())[0]
                search_utils.render_color_bar_figure(db_dict['color'], temp_dir + 'color{}.jpg'.format(i + 1))
                search_utils.render_texture_bar_figure(db_dict['texture'], temp_dir + 'texture{}.jpg'.format(i + 1))
                cv2.imwrite(temp_dir + 'shape{}.jpg'.format(i + 1), db_dict['shape'])
                od_temp, _ = search_utils.ObjDetect(result[i])
                cv2.imwrite(temp_dir + 'position{}.jpg'.format(i + 1), od_temp)

            # ===============================

            # Generate the json
            # ===============================
            # 0
            return_message['rawImage'] = usr_img_pre_dir  # the raw path
            # 1
            return_message['recallRatio'] = 'FOO'  # "92.5%" 查全率
            # 2
            return_message['precision'] = 'FOO'  # "87.5%" 查准率
            # 3
            return_message['position'] = temp_dir + 'usr_objd.jpg'  # path of the target marked image
            # 4
            features = dict()
            features['color'] = temp_dir + 'usr_img_color.jpg'  # path of the generated image
            features['texture'] = temp_dir + 'usr_img_texture.jpg'  # path of the generated image
            features['shape'] = temp_dir + 'usr_img_shape.jpg'  # path of the generated image
            return_message['features'] = features

            # 5
            results = dict()
            # 5.1
            one = dict()
            one['similarity'] = score[0]
            one['path'] = temp_dir + 'res1.jpg'
            one_features = dict()
            one_features['color'] = temp_dir + 'color1.jpg'
            one_features['texture'] = temp_dir + 'texture1.jpg'
            one_features['shape'] = temp_dir + 'shape1.jpg'
            one_features['position'] = temp_dir + 'position1.jpg'
            one['features'] = one_features
            # 5.2
            two = dict()
            two['similarity'] = score[1]
            two['path'] = temp_dir + 'res2.jpg'
            two_features = dict()
            two_features['color'] = temp_dir + 'color2.jpg'
            two_features['texture'] = temp_dir + 'texture2.jpg'
            two_features['shape'] = temp_dir + 'shape2.jpg'
            two_features['position'] = temp_dir + 'position2.jpg'
            two['features'] = two_features
            # 5.3
            three = dict()
            three['similarity'] = score[2]
            three['path'] = temp_dir + 'res1.jpg'
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
            # Try to make the temp folder
            try:
                os.mkdir(temp_dir)
            except Exception as x:
                print(str(x))

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
                raise x
            # ===============================

        except Exception as e:
            return HttpResponseBadRequest(str(e))
    else:
        return HttpResponseBadRequest('Not a POST request.')

    # Return http status 200
    return HttpResponse()
