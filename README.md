### FER_ROBOTICS_CW2_PDE4433
### FER_prototype by : M Dileep Kumar
#
# Facial Expressions Recognition with CNN and OpenCV

# Project Video : [Facial Expression Recognition](https://youtu.be/zLqIbR-Fc2I)


## Introduction

The "pre-processing.py" script was used to preprocess the dataset, which was obtained from Kaggle website in the form of a CSV file named "FER13.csv". The "haarcascade" folder contains the pre-trained Haar cascades used for face detection in the "FER_main_test.py" script.

To run the project, make sure all the necessary dependencies are installed, including OpenCV and Keras. Then, simply run the "FER_main_test.py" script, which will start the camera and detect facial expressions in real-time using the trained CNN model.

Note that the script will use the default camera on the device, but this can be changed by modifying the "cv2.VideoCapture()" line in the script. Also, ensure that the trained model files, "emotinal_model.h5" and "emotinal_model.json", are located in the same directory as the "FER_main_test.py" script.

#

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
* python FER_main_test.py*
```


5 . The program will start the camera and show a live video stream with a rectangle around the detected faces and the predicted facial expressions.
#

### File Structure

This repository contains the following files:

* *Model* folder: This folder contains the pre-trained CNN model saved in .h5 and .json format.

* *haarcascade* folder: This folder contains the pre-trained haarcascade classifier for face detection.

* *CNN_model.py*: This file contains the code to generate the pre-trained CNN model and save it in .h5 and .json format.

* *FER_main_test.py*: This file contains the main code for the FER prototype, which starts the camera, detects expressions using the pre-trained CNN model, and displays the recognized expression on the screen.

* *pre-processing.py*: This file contains the code to generate the data and labels into numpy format.


#
### Customization

If you wish to train your own CNN model, you can modify the "CNN_model.py" script to generate a new model. You can also customize the pre-processing steps by modifying the "pre-processing.py" script to fit your dataset.

### Summary of the CNN Model

```
dileep@Dileeps-MBP Jupyter_projects % /opt/homebrew/bin/python3 /Users/dileep/Desktop/Temp/Jupyter_projects/Emotion_detection
_with_CNN-main/FER_ROBOTICS_CW2/CNN_model.py
/opt/homebrew/lib/python3.10/site-packages/keras/optimizers/optimizer_v2/adam.py:117: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super().__init__(name, **kwargs)
2023-04-19 16:57:56.004399: W tensorflow/tsl/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/50
455/455 [==============================] - 304s 667ms/step - loss: 1.9887 - accuracy: 0.2131 - val_loss: 1.8259 - val_accuracy: 0.2591
Epoch 2/50
455/455 [==============================] - 298s 655ms/step - loss: 1.8241 - accuracy: 0.2531 - val_loss: 1.7144 - val_accuracy: 0.3059
Epoch 3/50
455/455 [==============================] - 301s 661ms/step - loss: 1.7180 - accuracy: 0.3039 - val_loss: 1.5680 - val_accuracy: 0.3505
Epoch 4/50
455/455 [==============================] - 305s 671ms/step - loss: 1.5949 - accuracy: 0.3686 - val_loss: 1.4914 - val_accuracy: 0.4211
Epoch 5/50
455/455 [==============================] - 299s 657ms/step - loss: 1.4987 - accuracy: 0.4114 - val_loss: 1.3871 - val_accuracy: 0.4303
Epoch 6/50
455/455 [==============================] - 303s 665ms/step - loss: 1.4473 - accuracy: 0.4382 - val_loss: 1.3571 - val_accuracy: 0.4700
Epoch 7/50
455/455 [==============================] - 302s 663ms/step - loss: 1.3976 - accuracy: 0.4633 - val_loss: 1.3275 - val_accuracy: 0.4728
Epoch 8/50
455/455 [==============================] - 301s 661ms/step - loss: 1.3610 - accuracy: 0.4823 - val_loss: 1.2888 - val_accuracy: 0.5146
Epoch 9/50
455/455 [==============================] - 303s 666ms/step - loss: 1.3220 - accuracy: 0.5013 - val_loss: 1.2321 - val_accuracy: 0.5328
Epoch 10/50
455/455 [==============================] - 303s 667ms/step - loss: 1.3005 - accuracy: 0.5131 - val_loss: 1.2226 - val_accuracy: 0.5378
Epoch 11/50
455/455 [==============================] - 300s 659ms/step - loss: 1.2785 - accuracy: 0.5250 - val_loss: 1.1663 - val_accuracy: 0.5622
Epoch 12/50
455/455 [==============================] - 303s 665ms/step - loss: 1.2489 - accuracy: 0.5369 - val_loss: 1.1778 - val_accuracy: 0.5607
Epoch 13/50
455/455 [==============================] - 303s 666ms/step - loss: 1.2221 - accuracy: 0.5476 - val_loss: 1.1364 - val_accuracy: 0.5693
Epoch 14/50
455/455 [==============================] - 300s 660ms/step - loss: 1.1955 - accuracy: 0.5583 - val_loss: 1.1439 - val_accuracy: 0.5802
Epoch 15/50
455/455 [==============================] - 300s 660ms/step - loss: 1.1800 - accuracy: 0.5639 - val_loss: 1.1189 - val_accuracy: 0.5842
Epoch 16/50
455/455 [==============================] - 299s 658ms/step - loss: 1.1578 - accuracy: 0.5716 - val_loss: 1.1087 - val_accuracy: 0.5960
Epoch 17/50
455/455 [==============================] - 300s 658ms/step - loss: 1.1319 - accuracy: 0.5861 - val_loss: 1.1096 - val_accuracy: 0.5932
Epoch 18/50
455/455 [==============================] - 300s 660ms/step - loss: 1.1047 - accuracy: 0.5917 - val_loss: 1.1041 - val_accuracy: 0.5923
Epoch 19/50
455/455 [==============================] - 300s 658ms/step - loss: 1.0986 - accuracy: 0.6021 - val_loss: 1.0649 - val_accuracy: 0.6121
Epoch 20/50
455/455 [==============================] - 300s 660ms/step - loss: 1.0724 - accuracy: 0.6116 - val_loss: 1.0668 - val_accuracy: 0.6136
Epoch 21/50
455/455 [==============================] - 305s 670ms/step - loss: 1.0518 - accuracy: 0.6190 - val_loss: 1.0833 - val_accuracy: 0.6084
Epoch 22/50
455/455 [==============================] - 302s 663ms/step - loss: 1.0375 - accuracy: 0.6245 - val_loss: 1.0299 - val_accuracy: 0.6232
Epoch 23/50
455/455 [==============================] - 300s 660ms/step - loss: 1.0136 - accuracy: 0.6322 - val_loss: 1.0341 - val_accuracy: 0.6344
Epoch 24/50
455/455 [==============================] - 299s 658ms/step - loss: 0.9900 - accuracy: 0.6404 - val_loss: 1.0278 - val_accuracy: 0.6170
Epoch 25/50
455/455 [==============================] - 301s 661ms/step - loss: 0.9691 - accuracy: 0.6482 - val_loss: 1.0108 - val_accuracy: 0.6368
Epoch 26/50
455/455 [==============================] - 300s 659ms/step - loss: 0.9538 - accuracy: 0.6560 - val_loss: 0.9975 - val_accuracy: 0.6294
Epoch 27/50
455/455 [==============================] - 299s 658ms/step - loss: 0.9404 - accuracy: 0.6661 - val_loss: 1.0234 - val_accuracy: 0.6353
Epoch 28/50
455/455 [==============================] - 299s 657ms/step - loss: 0.9180 - accuracy: 0.6698 - val_loss: 1.0117 - val_accuracy: 0.6356
Epoch 29/50
455/455 [==============================] - 304s 669ms/step - loss: 0.8974 - accuracy: 0.6778 - val_loss: 1.0078 - val_accuracy: 0.6350
Epoch 30/50
455/455 [==============================] - 305s 671ms/step - loss: 0.8880 - accuracy: 0.6817 - val_loss: 1.0376 - val_accuracy: 0.6384
Epoch 31/50
455/455 [==============================] - 300s 660ms/step - loss: 0.8680 - accuracy: 0.6871 - val_loss: 0.9966 - val_accuracy: 0.6458
Epoch 32/50
455/455 [==============================] - 301s 662ms/step - loss: 0.8663 - accuracy: 0.6925 - val_loss: 0.9851 - val_accuracy: 0.6437
Epoch 33/50
455/455 [==============================] - 304s 668ms/step - loss: 0.8380 - accuracy: 0.7001 - val_loss: 1.0004 - val_accuracy: 0.6545
Epoch 34/50
455/455 [==============================] - 304s 667ms/step - loss: 0.8181 - accuracy: 0.7053 - val_loss: 0.9838 - val_accuracy: 0.6554
Epoch 35/50
455/455 [==============================] - 302s 664ms/step - loss: 0.8008 - accuracy: 0.7137 - val_loss: 0.9832 - val_accuracy: 0.6567
Epoch 36/50
455/455 [==============================] - 300s 660ms/step - loss: 0.7996 - accuracy: 0.7197 - val_loss: 0.9929 - val_accuracy: 0.6591
Epoch 37/50
455/455 [==============================] - 300s 660ms/step - loss: 0.7789 - accuracy: 0.7252 - val_loss: 1.0344 - val_accuracy: 0.6502
Epoch 38/50
455/455 [==============================] - 309s 679ms/step - loss: 0.7615 - accuracy: 0.7322 - val_loss: 1.0241 - val_accuracy: 0.6628
Epoch 39/50
455/455 [==============================] - 303s 666ms/step - loss: 0.7540 - accuracy: 0.7345 - val_loss: 1.0007 - val_accuracy: 0.6551
Epoch 40/50
455/455 [==============================] - 301s 662ms/step - loss: 0.7310 - accuracy: 0.7450 - val_loss: 1.0125 - val_accuracy: 0.6622
Epoch 41/50
455/455 [==============================] - 302s 663ms/step - loss: 0.7175 - accuracy: 0.7489 - val_loss: 1.0304 - val_accuracy: 0.6486
Epoch 42/50
455/455 [==============================] - 300s 660ms/step - loss: 0.7177 - accuracy: 0.7496 - val_loss: 1.0273 - val_accuracy: 0.6529
Epoch 43/50
455/455 [==============================] - 301s 661ms/step - loss: 0.6982 - accuracy: 0.7567 - val_loss: 1.0486 - val_accuracy: 0.6591
Epoch 44/50
455/455 [==============================] - 299s 658ms/step - loss: 0.6861 - accuracy: 0.7627 - val_loss: 1.0318 - val_accuracy: 0.6591
Epoch 45/50
455/455 [==============================] - 302s 663ms/step - loss: 0.6767 - accuracy: 0.7662 - val_loss: 1.0426 - val_accuracy: 0.6613
Epoch 46/50
455/455 [==============================] - 299s 658ms/step - loss: 0.6676 - accuracy: 0.7685 - val_loss: 1.0715 - val_accuracy: 0.6632
Epoch 47/50
455/455 [==============================] - 300s 658ms/step - loss: 0.6524 - accuracy: 0.7732 - val_loss: 1.0635 - val_accuracy: 0.6610
Epoch 48/50
455/455 [==============================] - 301s 661ms/step - loss: 0.6462 - accuracy: 0.7765 - val_loss: 1.0470 - val_accuracy: 0.6607
Epoch 49/50
455/455 [==============================] - 299s 657ms/step - loss: 0.6255 - accuracy: 0.7834 - val_loss: 1.0362 - val_accuracy: 0.6632
Epoch 50/50
455/455 [==============================] - 302s 665ms/step - loss: 0.6268 - accuracy: 0.7879 - val_loss: 1.1057 - val_accuracy: 0.6551
Saved model to disk
dileep@Dileeps-MBP Jupyter_projects %

```


### Acknowledgments


This project was developed by M Dileep Kumar as part of [PDE4433/Middlesex University Dubai]. Special thanks to Dr. Maha Saadeh for their guidance and support.

### Contact

If you have any questions or comments about this project