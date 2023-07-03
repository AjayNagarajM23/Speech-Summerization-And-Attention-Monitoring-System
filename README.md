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

The Speech Summarization and Attention Monitoring System for Classroom Assistance is a comprehensive project that aims to provide valuable insights into classroom interactions and student engagement. By analyzing attention levels and generating summarized transcripts, this
