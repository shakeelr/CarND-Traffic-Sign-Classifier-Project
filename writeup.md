# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/signdatahistogram.png "Histogram of Sign Types"
[image2]: ./writeup/rawsign.png "Raw Sign Image"
[image3]: ./writeup/preprocessedsign.png "Sign After Pre Processing"
[image4]: ./writeup/lenetvsdeeplenet.png "LeNet vs DeepLeNet Training"
[image5]: ./writeup/newsigns.png "New Signs"
[image6]: ./writeup/newsignspredictionstop5.png "New Signs with Top 5 Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shakeelr/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Summary statistics of the traffic signs data set were calculated using the len(), np.shape(), and np.unique() methods as follows:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It contains a histogram of the data in the train, validation, and test data sets.  Note that there are significantly more examples of some sign types compared to others in each of the train, validation, and test datasets, which might cause some training challenges.  However, the distribution of sign types is similar between the train, validation, and test datasets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

A number of preprocessing steps were taken.  First, the images were converted to greyscale then histogram equalized to give the images a similar level of contrast.  The images were also normalized by dividing each pixel by 128 and zero centered by subtracting the mean from each pixel.

Below is an example of a traffic sign prior to pre processing...

![alt text][image2]

And after pre processing...

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried two different models to predict traffic signs.  The first was a standard LeNet model, and the second was based on the LeNet architecture but contained an extra convolutional layer, dropout layers, a larger number of feature maps, and larger kernel sized.  I refer the the second model as DeepLeNet.

##### LeNet Model:

| Layer					|Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 Grayscale image   					| 
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 14x14x6 					|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 5x5x16 					|
| Flatten				| Outputs 400									|
| Fully connected		| Outputs 120  									|
| RELU					|												|
| Fully connected		| Outputs 84  									|
| RELU					|												|
| Fully connected		| Outputs 43  									|

##### DeepLeNet Model:
 
| Layer					|Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 Grayscale image   					| 
| Convolution 5x5		| 1x1 stride, same padding, outputs 32x32x16 	|
| Leaky RELU			| alpha = 0.2									|
| Dropout 				| keep probability = 0.66 						|
| Max pooling			| 2x2 stride, outputs 16x16x16 					|
| Convolution 5x5		| 1x1 stride, same padding, outputs 16x16x32	|
| Leaky RELU			| alpha = 0.2									|
| Dropout 				| keep probability = 0.66 						|
| Max pooling			| 2x2 stride, outputs 8x8x32 					|
| Convolution 3x3		| 1x1 stride, same padding, outputs 8x8x64		|
| Leaky RELU			| alpha = 0.2									|
| Dropout 				| keep probability = 0.66 						|
| Max pooling			| 2x2 stride, outputs 4x4x64 					|
| Flatten				| Outputs 1024									|
| Fully connected		| Outputs 128  									|
| Leaky RELU			| alpha = 0.2									|
| Dropout 				| keep probability = 0.66 						|
| Fully connected		| Outputs 84  									|
| Leaky RELU			| alpha = 0.2									|
| Dropout 				| keep probability = 0.66 						|
| Fully connected		| Outputs 43  									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a used an Adam Optimizer to minimize the mean reduced cross entropy loss function, which was applied to the softmax of the logits from the LeNet and DeepLeNet models.  Both the LeNet and DeepLeNet models were trained using a batch size of 256 and learning rate of 0.5.  The LeNet model was trained for 300 Epochs but showed evidence of overtraining with validation losses increasing within the first 20 Epochs.  The DeepLenet model was slower to train but ultimately was trained for 500 Epochs with no evidence of overtraining - likely thanks to the use of dropout layers.

A figure comparing the training between the LeNet and DeepLeNet models is shown below:

![alt text][image4]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


The LeNet model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.938 
* test set accuracy of 0.916

The DeepLeNet model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.952 
* test set accuracy of 0.934

I started with the LeNet model, and tried different preprocessing techniques (grayscale, historgram equalization, normalization, and zero centering) until I could get a validation accuracy above 0.93.  However, it appeared that the LeNet model was overtraining, as seen by the increasing validation losses and reflected in test accuracy of only 0.916.  At this point I decided try some changes to the LeNet architecture to try and get a higher accuracy without overtraining.  I added an additional convolutional layer, increased the number of feature maps, used Leaky ReLU activation functions to prevent potential dying ReLU problems, and added dropout layers to prevent overfitting.  With these archtectural changes I achieved a validation accuracy of 0.952 and a test accuracy of 0.934 with less evidence of overfitting,  even after 500 epochs.  A weakness of the "DeepLeNet" architecture was that it trained relatively slowly.  Alternative models such as AdamNet or GoogLeNet might achieve similar or better accuracy with less training, however I was happy with my results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, the images were cropped and resized to 32x32 to match the images in the German traffic sign dataset:

![alt text][image4]

The first image will likely be the most difficult to classify.  There are a number of very similar speed limit signs in the German traffic sign dataset, and differentiating them might be a challange for the model.  The third image might also be slighly difficult to classify since the sign is rotated slightly, and I did not augment my dataset with rotated signs.  I'd expect the second, forth and fifth image to be relatively easy to classify, as the priority road, yield, and stop signs are fairly unique in it's shape and color compared to the other signs in the dataset.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image					| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30 		| Speed Limit 30								| 
| Priority Road			| Priority Road									|
| No Entry				| No Entry						 				|
| Yield					| Yield											|
| Stop Sign				| Stop Sign										|


The model was able to correctly guess all of the traffic signs, or 100% accuracy.  The accuracy was higher than on the test set, although the same size of new images is small.  I also feel that it's important to note that the images were all clearly taken stock photos that I had cropped and resized, so the model may have had a relatively easy time classifying them.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The five new signs, along with their predictions and top5 probabilities can be seen in the image below:

![alt text][image5] 

The top 5 probabilities for the first image (Speed Limit 30) are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .71         			| Speed Limit 30								| 
| .21     				| Speed Limit 70								|
| .03					| Stop											|
| .03	      			| Speed Limit 120				 				|
| .01				    | Keep Left      								|


The top 5 probabilities for the second image (Priority Road) are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority Road									| 
| .00     				| Yield 										|
| .00					| Ahead Only									|
| .00	      			| No Entry						 				|
| .00				    | Go Straight or Right 							|


The top 5 probabilities for the third image (No Entry) are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No Entry   									| 
| .00     				| Stop 											|
| .00					| Turn Right Ahead								|
| .00	      			| No Passing					 				|
| .00				    | Vehicles over 3.5 tons prohibited				|

The top 5 probabilities for the fourth image (Yield) are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield   										| 
| .00     				| Speed Limit 60								|
| .00					| Speed Limit 80								|
| .00	      			| Road Work						 				|
| .00				    | Priority Road      							|

The top 5 probabilities for the fifth image (Stop) are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop sign   									| 
| .00     				| Turn right ahead								|
| .00					| No Entry										|
| .00	      			| Yield							 				|
| .00				    | Speed limit (120km/h)     					|

Other then the first image (Speed Limit 30), the model was very confident in it's predictions, with all other signs predicting the correct sign with nearly 100% probability.  I expected the Speed Limit 30 sign to possibly get confused with some of the other speed limit signs, and this was seen in the top 5 probabilities.  However the top 5 probabilities of the first sign also included the stop and keep left signs, which weren't expected.
