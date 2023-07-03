from django.contrib import admin
from django.urls import path, include
from app import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload', views.upload, name='upload'),
    path('result', views.result, name='result'),
    path('summary', views.summary, name='summary'),    
]