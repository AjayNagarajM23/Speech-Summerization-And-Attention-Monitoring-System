from django.shortcuts import render
from .models import vid
from django.shortcuts import render, redirect
from django.contrib import messages
from .processing import final
from .processing import cal_summary
from .processing import save_audio
# Create your views here.

def index(request):
    return render(request,"index.html")

def upload(request):
    test = None
    if request.method == "POST":
        temp_video = request.FILES['video']
        new_vid = vid(video = temp_video)
        new_vid.save()
        save_audio()
        messages.success(request, "New Video Added")
        
    return render(request, "upload.html")

def result(request):
    result = final()
    return render(request, 'result.html',{'result':result})



def summary(request):
    text, summary = cal_summary()
    return render(request, 'summary.html', {'text':text, 'summary':summary})