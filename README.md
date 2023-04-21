### FER_ROBOTICS_CW2_PDE4433
### FER_prototype by : M Dileep Kumar
#
# Facial Expressions Recognition with CNN and OpenCV

# Project Video : [Facial Expression Recognition](https://youtu.be/zLqIbR-Fc2I)


## Introduction

This project aims to detect facial expressions in real-time using a Convolutional Neural Network (CNN) and OpenCV. It includes a trained CNN model that was generated using the "CNN_model.py" script, and a script to perform facial expressions recognition in real-time named "FER_main_test.py".

### Getting Started

To run the facial expressions recognition program, follow these steps:

1 . Install the required dependencies:
* Python 3.6 or higher
* OpenCV 4.5 or higher
* NumPy 1.19 or higher
* Keras 2.4 or higher
* TensorFlow 2.4 or higher

2 . Download or clone the repository to your local machine.

3 . Navigate to the project directory using a terminal or command prompt.

4 . Run the following command to start the program:

```
python FER_main_test.py
```


5 . The program will start the camera and show a live video stream with a rectangle around the detected faces and the predicted facial expressions.
#

### File Structure

This repository contains the following files:

* Model folder: This folder contains the pre-trained CNN model saved in .h5 and .json format.

* haarcascade folder: This folder contains the pre-trained haarcascade classifier for face detection.

* CNN_model.py: This file contains the code to generate the pre-trained CNN model and save it in .h5 and .json format.

* FER_main_test.py: This file contains the main code for the FER prototype, which starts the camera, detects expressions using the pre-trained CNN model, and displays the recognized expression on the screen.

* pre-processing.py: This file contains the code to generate the data and labels into numpy format.


#
### Customization

If you wish to train your own CNN model, you can modify the "CNN_model.py" script to generate a new model. You can also customize the pre-processing steps by modifying the "pre-processing.py" script to fit your dataset.


### Acknowledgments


This project was developed by M Dileep Kumar as part of [PDE4433/Middlesex University Dubai]. Special thanks to Dr. Maha Saadeh for their guidance and support.

### Contact

If you have any questions or comments about this project