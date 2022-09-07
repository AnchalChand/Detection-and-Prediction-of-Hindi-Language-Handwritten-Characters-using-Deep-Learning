# DETECTION-AND-PREDICTION-OF-HINDI-LANGUAGE-HANDWRITTEN-CHARACTERS-USING-DEEP-LEARNING

# ABSTRACT
• This project is developed to detect the Hindi handwritten characters from the input image and is implemented in Python programming language. This project uses the concept of Deep Learning. The four-layered Convolutional Neural Network is used for this project. The dataset used to train the model contains 92,000 images of handwritten Hindi characters belonging to 46 classes with the image size of 32x32 in .png format.

# INTRODUCTION
• Handwritten character recognition is a buzzing term when working in the field of artificial intelligence, embedded with computer vision. This is mainly because of its important applications like converting handwritten data into digitized form for easy modifications as per the requirements, automatic data entry of documents, human and robot interaction and much more. Handwritten character recognition helps the blind and visually impaired users as it is a preliminary approach in the process to convert data into audio form. The handwriting recognition algorithms detect and recognize the text from images/videos and digitize them by converting into a machine-readable form. \
• This project, titled as “Detection of Hindi language handwritten characters using deep learning”, works on Convolutional Neural Network (CNN) to find and recognize the handwritten characters of Hindi language.

# MOTIVATION
• The detection of handwritten characters is an important step in the process to digitize the handwritten data, which has its vital applications indeed. It provides the features to modify and store that data in required format which is not possible with handwritten text. Also, real-time detection of handwritten characters in Hindi language can be helpful to read important information. \
• These features have motivated me to opt for this project. Working on this project gave me an opportunity to develop the mechanism for the same and to explore more in the field of Python programming language and Deep Learning features. 

# METHODOLOGY FOLLOWED
• A four layered convolutional neural network is used for this project. \
• A typical CNN has the following 4 layers:
  1. Input layer
  2. Convolution layer
  3. Pooling layer
  4. Fully connected layer
![image](https://user-images.githubusercontent.com/82054687/188876071-74be4b58-8110-44ce-a989-bc6d58c4a2dd.png)
 
 ### Dataset
•	The dataset contains 92,000 images of handwritten Hindi characters belonging to 46 classes. The images are of size 32x32 in .png format.
 ### Architecture
•	The architecture comprises of four CNN layers.
 ### Implementation Details
•	Loss function- Categorical Cross entropy Optimizer- Adam Final output layer activation- Softmax
 ### Output
•	Accuracy achieved in 25 epochs- 96.96% \
•	Accuracy achieved in 40 epochs- 97.54%

# WORKING
• The method used in this project is divided into following steps: 
  1. To collect the data of the Hindi characters from the dataset used. 
  2. To pre-process the input image, check null and missing values and use the normalization technique (Normalization is a data pre-processing tool used to bring the numerical data to a common scale without distorting its shape). 
  3. To detect the presence of handwritten characters in Hindi language. 
  4. To extract the features using Deep learning algorithm - Convolutional Neural Network (CNN) for recognition of handwritten character system. 
  5. To apply an optimization technique, Adaptive Moment (Adam) Estimation for promising result. 
  
• The deep learning techniques are basically composed of multiple hidden layers, and each hidden layer consists of multiple neurons, which computes the suitable weights for the deep network. A convolutional neural network has multiple convolutional layers to extract the features automatically. In the case of deep learning models, multiple convolutional layers have been adopted to extract discriminating features multiple times. This is one of the reasons that deep learning models are generally successful.



### Input Image

![input image](https://user-images.githubusercontent.com/82054687/188880219-d7d74e1d-6fcd-4e02-be46-cd14e428fdee.png)


### Output Image

![output image](https://user-images.githubusercontent.com/82054687/188879969-56090945-4405-4b10-bd72-be4da18236b2.png)



# FUTURE GOALS
• To improve the efficiency of model prediction. \
• To implement the project on real-time character detection. 

# REFERENCES
1.	https://www.wikipedia.org
2.	https://www.geeksforgeeks.org
3.	https://www.youtube.com 
4.	https://towardsdatascience.com
