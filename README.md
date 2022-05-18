# Facial_Emotion_Recognition

Facial Emotion Recognition (FER) is the technology that analyses facial expressions from both
static images and videos in order to reveal information on one's emotional state. It classifies
images of human faces into emotion categories using Deep Neural Networks.[1] Deep Learning
(DL) based emotion detection gives performance better than traditional methods with image
processing.Deep learning extracts unique facial embeddings from images of faces and uses a
trained model to recognize photos from a database in other photos and videos.[2]
It comprises of mainly three steps :
● Face Detection
● Facial Feature Extraction
● Expression classification to an emotional state.

In this proposed model various components of deep neural learning came together to provide a
remarkable result. Initially an image of 48 X 48 pixels, gray scale in nature has been passed to
the proposed architecture. This image then conceived by a pair of convolutional layer of 64
filters each in order to identify the minutes features of the input image. The feature map that
came after the last step goes through batch normalization which basically standardizes the
input coming from the last layer and smooth the training process. The output of the batch
normalization layer goes through the relu activation function whose purpose is to introduce
non-linearity in the model.Then output of this layer passes through the max pooling layer which
extracts out the most dominant features . Max Pooling layer reduces the dimension and also
decrease the computational complexity.
This all process occur in blocks. We have such 3 blocks which differ only in the number of filters
in the convolutional layer which is 128 and 256 respectively in block 2 and block 3.After
completing the task of feature extraction our model enters into the classification part. In the
classification part, 3 dense layers have been used where 2 dense layers consist of 128 neurons
and one has 7 neutrons.This layers mostly intended to classify the expression which image
contains and separate out. At last the softmax layer has been used to give the result in a
multinomial probability function to find out which expression has more characteristic in the input
image. In the output layer one expression appears telling the expression of the input image.

![Drawing sketchpad (1)](https://user-images.githubusercontent.com/71483319/169095921-b456e795-164a-4551-8ab9-a91e14a360cc.jpeg)

Convolutional layer: We used different sizes of kernels to extract out the feature that aids our
model to identify the various emotion states.Used to extract feature map. Kernels of size 3 X 3,
1 X 1 are being used in our model.

Batch Normalization: Batch normalization is a powerful regularization technique that decreases
training time and improves performance by addressing internal covariate shift that occurs during
training.

Activation Function: An activation function in a neural network defines how the weighted sum of
the input is transformed into an output from a node or nodes in a layer of the network. Used
ReLu in hidden layers and Softmax in the output layer.

Max Pooling: Pooling layers are used to reduce the dimensions of the feature maps. Thus, it
reduces the number of parameters to learn and the amount of computation performed in the
network. Used Max Pooling in the architecture.

Flattening: Converts the 2D matrix coming from feature extractor block to 1D array.

Dense Layer: Utilizes the output from the convolution process and predicts the class. 3 Fully
connected dense layers with 128, 128 ,7 neurons each are used.

Dropout: It is a regularization method that approximates training a large number of neural
networks with different architectures in parallel.This technique reduces overfitting. Uses a
combination of 0.1 and 0.2 dropout rate .

Loss Function and optimizer: Loss Function helps in adjusting the weights of neurons during
backpropagation. Optimizers are algorithms used to change the attributes such as weights and
learning rate in order to reduce the losses. Used Categorical Cross entropy as loss function and
adam optimizer.

Softmax Layer: Outputs a vector of values that sum to 1.0. It is related to the argmax function
that outputs a 0 for all options and 1 for the chosen option and Classifies into 1 of 7 emotion
categories
