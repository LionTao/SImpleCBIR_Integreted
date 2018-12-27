from django.shortcuts import render

# Create your views here.

from django.http import Http404
from main.makecache import cacheAll


def index(request):
    raise Http404("你来到了知识的荒原...")

def initData(request):
    cacheAll()
