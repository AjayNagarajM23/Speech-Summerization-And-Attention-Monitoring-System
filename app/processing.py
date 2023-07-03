"""
Attention Monitoring imports
---------------------------------------------------------------------------------
"""
import torch
import numpy as np  
import cv2
from yolov5.detect import run
import os
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import shutil
import time
path = 'images/crops/Focused/'
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

"""
Speech Summerization imports
----------------------------------------------------------------------------------
"""
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from transformers import pipeline
import moviepy.editor
"""
INDIVIDUAL ATTENTION MONITORING
----------------------------------------------------------------------------------
"""
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))


# Saving cropped Images
def save_crops(image):
    prGreen("******* CROPPING IMAGES *******")
    run(weights='best.pt',
        hide_labels=True,
        save_crop=True,
        source=image,
        name='images',
        project='',
        nosave=True,
        classes=15)

  
# Individual face recognition
def face_reco():
    prGreen("******* FACE RECOGNITION *******")

    resdict = {}
    final_res = []
    ResultMap = {}
    final_detect = []

    # final_detect = np.array(final_detect)
    with open("ResultsMap_final.pkl", 'rb') as f:
        ResultMap = pickle.load(f)
    subject = os.listdir(path)
    pre_process(subject)
    samples = os.listdir(path)

    test_arr = []

    for sample in samples:
        img = image.load_img(path+sample, target_size = (224, 224))
        test_arr.append(image.img_to_array(img))

    test_arr = np.array(test_arr)
    result = load_reco_model(test_arr)
    for res in result:
        final_detect.append(ResultMap[np.argmax(res)])
    #print(final_detect)

    a = np.unique(final_detect, return_counts=True)
    shutil.rmtree('images')
    files = os.listdir('frames')
    total = len(files)

    for i in range(0, len(a[1])):
        per = (int(a[1][i])/total)*100
        a[1][i] = per

    for file in files: os.remove("frames/"+file)
    
    for i in range(len(a[0])):
        resdict = {'id': a[0][i], 'attention': a[1][i]}
        final_res.append(resdict)

    print(final_res)
    os.remove('media/videos/test_vid.mp4')
    return final_res


# Loading the Face Recognition Model
def load_reco_model(imgs):
    model = load_model("final _08.h5")
    result = model.predict(imgs)
    return result


# Pre-processing Of Images
def pre_process(imgs):
    prGreen("******* PRE-PROCESSING IMAGES*******")
    for img in imgs:
        img_clr = cv2.imread(path+img)
        img_gry = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)

        img_clr = np.array(img_clr)
        img_gry = np.array(img_gry)
    
        faces = facecascade.detectMultiScale(
               img_gry, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) != 1:
            os.remove(path+img)
            continue

        for (x_, y_, w, h) in faces:
            face_detect = cv2.rectangle(img_clr,
                                        (x_, y_),
                                        (x_+w, y_+h),
                                        (255, 0, 255), 2)

            size = (224, 224)

            roi = img_clr[y_: y_ + h, x_: x_ + w]

            resized_image = cv2.resize(roi, size)
            image_array = np.array(resized_image)
            os.remove(path+img)
            cv2.imwrite(path+img, image_array)


# Saving the Frames in a Video at rate 1FPS
def save_frame():
    prGreen("******* SAVING VIDEO FRAMES *******")
    KPS = 1
    VIDEO_PATH = "media/videos/test_vid.mp4"
    IMAGE_PATH = "/home/ajaynagarajm/Major_Final/frames/"
    EXTENSION = ".png"
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    hop = round(fps / KPS)
    curr_frame = 0
    while(True):
        ret, frame = cap.read()
        if not ret: break
        if curr_frame % hop == 0:
            name = IMAGE_PATH + "_" + str(curr_frame) + EXTENSION
            cv2.imwrite(name, frame)
        curr_frame += 1
    cap.release()


# Calling all above for Attention Monitoring
def final():
    prGreen("*******  STARTING  *******")
    save_frame()
    time.sleep(10)
    save_crops("frames/")
    time.sleep(10)
    result = face_reco()
    return result


"""
SPEECH SUMMERIZATION
-----------------------------------------------------------------------------
"""
# Speech Summerization
def cal_summary():
    prGreen("******* PREPARING SUMMARY *******")
    FRAME_RATE = 16000
    CHANNELS=1
    model = Model(model_name="vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, FRAME_RATE)
    rec.SetWords(True)

    mp3 = AudioSegment.from_mp3("media/audios/sample.mp3")
    mp3 = mp3.set_channels(CHANNELS)
    mp3 = mp3.set_frame_rate(FRAME_RATE)

    rec.AcceptWaveform(mp3.raw_data)
    result = rec.Result()

    import json
    text = json.loads(result)["text"]

    summarizer = pipeline("summarization")
    split_tokens = text.split(" ")
    docs = []
    for i in range(0, len(split_tokens), 850):
        selection = " ".join(split_tokens[i:(i+850)])
        docs.append(selection)

    summaries = summarizer(docs)
    final_summary = summaries[0]['summary_text']
    os.remove("media/audios/sample.mp3")
    return text, final_summary


"""
---------------------------------------------------------------------------------------
"""
# Saving Audio From the Video
def save_audio():
    prGreen("******* AUDIO SAVED*******")
    video = 'media/videos/test_vid.mp4'
    video = moviepy.editor.VideoFileClip(video)
    audio = video.audio
    audio.write_audiofile("media/audios/sample.mp3")