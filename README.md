# Speech Summarization and Attention Monitoring System for Classroom Assistance

The Speech Summarization and Attention Monitoring System for Classroom Assistance is an innovative project aimed at providing a comprehensive solution for monitoring and summarizing classroom interactions. The primary goal of this system is to enhance the learning experience by analyzing attention levels and generating summarized transcripts of classroom discussions.

## Project Overview

This project report outlines the various steps involved in developing this system, including data set collection, labeling, model training, and implementation of speech-to-text and text summarizing techniques.

### Data Set Collection and Labeling

The initial phase of the project involved collecting a data set of 300 images featuring people in a classroom setting. These images were obtained to prepare a robust data set for training and evaluation purposes. To facilitate the labeling process, the `labelmeimg` module of Python was utilized, following a YOLO labeling scheme. The data set was categorized into two classes: Focused and Not Focused. A `dataset.yaml` file was created, which would later be used for training the YOLOv5 model.

### Model Training and Selection

To determine the optimal model for object detection, a comparative study was conducted between the YOLOv5s and YOLOv5l6 models. It was observed that while YOLOv5s provided faster results, it lacked accuracy when multiple faces were involved. On the other hand, YOLOv5l6 exhibited better accuracy but was relatively slower. Consequently, the decision was made to proceed with training the YOLOv5l6 model. The training process involved 300 epochs with a batch size of 16 and an image size of 1080. The trained model achieved impressive performance, with a precision of 0.99 and a recall of 1.

### Face Recognition

Once the YOLOv5l6 model was trained, it was employed to detect and crop the focused faces in the images. This step aimed to isolate the facial regions of individuals who were actively engaged in the classroom. The cropped face images were saved in a separate folder, laying the groundwork for the subsequent face recognition phase. For face recognition, the VGGFace2 data set was downloaded, comprising 47,534 images belonging to 204 different classes. The images were pre-processed by converting them to grayscale and applying the haarcascade frontal face technique to crop the faces. The resulting data set, consisting of colored cropped face images, served as the foundation for training the face recognition model.

### Training and Testing

The data set was split into training and testing subsets, with 80 percent of the samples allocated for training and 20 percent for testing. A custom convolutional neural network (CNN) model was designed, comprising five convolutional layers and six fully connected layers. This model was trained for 75 epochs, resulting in an impressive accuracy rate of 98%. Additionally, a pre-trained model was used to achieve an accuracy of 85% by adding two additional dense layers to the model.

### Speech-to-Text and Text Summarization

To convert speech input into text, the VOSK model was employed, generating a transcript of the speech. This transcript was then tokenized and provided as input to the T5small model, which generated concise summaries of the transcriptions.

## Conclusion

In conclusion, the "Speech Summarization and Attention Monitoring System for Classroom Assistance" successfully integrated computer vision and natural language processing techniques to monitor attention levels in a classroom and provide summarized text
outputs. The project demonstrated effective insights into classroom dynamics and the
potential to support educational environments for enhanced learning experiences.


## Results
![Screenshot from 2023-06-03 10-57-26](https://github.com/AjayNagarajM23/Speech-Summerization-And-Attention-Monitoring-System/assets/105035860/2bc47e02-3536-4241-b20e-0cc70ae04642)
![Screenshot from 2023-06-03 10-58-54](https://github.com/AjayNagarajM23/Speech-Summerization-And-Attention-Monitoring-System/assets/105035860/17adebe6-83b7-4f6e-9f11-03d57dab23fe)
![Screenshot from 2023-06-03 10-59-21](https://github.com/AjayNagarajM23/Speech-Summerization-And-Attention-Monitoring-System/assets/105035860/9deeb965-79c2-480a-aefb-38a36a58ff13)



