from django.urls import path

from main import views
from main import upload_pic
from main import get_pic

urlpatterns = [
    path('', views.index, name='index'),
    path('start', upload_pic.upload_pic, name='upload_pic'),
    path('image', get_pic.get_pic, name='get_pic'),
    path('init',views.initData,name='initData')
]
