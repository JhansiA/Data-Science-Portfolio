# Capstone Project
## Proposal
## Domain Background
According to the CDC motor vehicle safety division, one in five car
accidents is caused by a distracted driver. Sadly, this translates to 425,000
people injured and 3,000 people killed by distracted driving every year.
State Farm hopes to improve these alarming statistics, and better insure their
customers, by testing whether dashboard cameras can automatically detect
drivers engaging in distracted behaviours.
## Problem Statement
Given a dataset of 2D dashboard camera images, an algorithm needs to be
developed to classify each driver's behaviour and determine if they are
driving attentively, wearing their seatbelt, or taking a selfie with their friends in
the backseat etc..? This can then be used to automatically detect drivers
engaging in distracted behaviours from dashboard cameras.
## Datasets and Inputs
driver images, each taken in a car with a driver doing something in the car
(texting, eating, talking on the phone, makeup, reaching behind, etc) were
provided.
The 10 classes to predict are:
- c0: safe driving
- c1: texting - right
- c2: talking on the phone - right
- c3: texting - left
- c4: talking on the phone - left
- c5: operating the radio
- c6: drinking
- c7: reaching behind
- c8: hair and makeup
- c9: talking to passenger

Following are the file descriptions and URL’s from which the data can be
obtained :
- imgs.zip - zipped folder of all (train/test) images
- sample_submission.csv - a sample submission file in the correct format
- driver_imgs_list.csv - a list of training images, their subject (driver) id,
and class id

https://www.kaggle.com/c/state-farm-distracted-driver-detection/download/
driver_imgs_list.csv.zip

https://www.kaggle.com/c/state-farm-distracted-driver-detection/download/
imgs.zip     

https://www.kaggle.com/c/state-farm-distracted-driver-detection/download/
sample_submission.csv.zip
## Solution Statement
A deep learning algorithm will be developed using Tensorflow/Keras and
will be trained with training data. Specifically a CNN will be implemented
in Tensorflow/Keras and will be optimised to minimize multi-class
logarithmic loss as defined in the Evaluation Metrics section. Predictions
will be made on the test data set and will be evaluated.
## Benchmark Model
The model with the Public Leaderboard score(multi-class logarithmic
loss) of 0.08690 will be used as a benchmark model. Attempt will be 
made so that score(multi-class logarithmic loss) obtained will be among
the top 50% of the Public Leaderboard submissions.
## Evaluation Metrics
Submissions are evaluated using the multi-class logarithmic loss. Each
image has been labeled with one true class. For each image, you must
submit a set of predicted probabilities (one for every image). The formula is
then,                                                                                       

logloss = − 1
N
N
∑
i=1
M
∑
j=1
yij
log(pij
),


where N is the number of images in the test set, M is the number of
image class labels, log is the natural logarithm, is 1 if observation
i belongs to class j and 0 otherwise, and is the predicted probability that
observation i belongs to class j.

The submitted probabilities for a given image are not required to sum to one
because they are rescaled prior to being scored (each row is divided by the
row sum). In order to avoid the extremes of the log function, predicted
probabilities are replaced with max(min(p, 1 − 10−15), 10−15)

## Project Design
From the description and problem statement it can be inferred that
computer vision can be used to arrive at a solution. CNN class of deep
learning algorithm can be employed for this problem.
Initially data exploration will be carried out to understand possible labels,
range of values for the image data and order of labels. This will help
preprocess the data and can end up with better predictions.
After this necessary preprocess functions will be implemented , data will
be randomised and CNN will be implemented in Tensorflow/Keras.
Finally necessary predictions on the test data will be carried out and
these will be evaluated.
