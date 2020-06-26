# Emojinator

> Real-time gestures detection as shown by the user, using CNN

## Modules used
 - OpenCV
 - NumPy
 - Pandas
 - Keras; Tensorflow
 - Scikit-learn

## Implementation
 1. `createGest.py` allows the user to input gestures by detecting contours
 2. `createCSV.py` converts these images into a CSV file for processing
 3. `train.py` trains the CNN model using relu activation function
 4. `application.py` performs detection on real-time input
 
## Points to note
 - 11 classes; each with 1200 gestures
 - Number of training samples: 7919
 - Number of test samples: 5280 - 30%
 - Trained over 10 epochs
 - Batch size: 64

*Observe the values for different parameters*

**Train accuracy -** 99.97%

**Test accuracy -**  99.85%

**CNN error -** 3.085%
