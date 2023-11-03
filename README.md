# Deep-Learning-Based-Road-Sign-Detection

### Abstract
With the rise of artificial intelligence (A.I.) and the advancements in technology, the need for
proper road sign recognition is important for aiding driver awareness of road signs. When considering
autonomous vehicles, it is vital for the vehicle to recognize different road signs to avoid accidents. The
goal of our project is to analyze various road signs and categorize them for autonomous vehicles to
recognize signs effectively to increase road safety. Traffic sign recognition (TSR) can improve road safety
by providing real-time information to drivers about their surroundings including speed limits, stop signs,
and road conditions.

### 1 Business Understanding
As artificial intelligence (A.I.) and technology continue to advance, it has become increasingly
crucial to ensure proper traffic sign recognition (TSR) to enhance driver awareness of speed limits and
other road signs. With the advent of autonomous vehicles the importance of accurate TSR cannot be
overstated, as low accuracy can lead to accidents. Our project aims to analyze various traffic signs and
traffic lights, categorize them, and develop an effective recognition system for autonomous vehicles.
Real-time TSR can provide drivers with vital information about their surroundings including speed limits,
stop signs, and road conditions, thereby contributing to improved road safety. Additionally, such systems
can positively impact driver behavior by providing real-time feedback on speed and compliance with traffic
laws. Given the projected rise in self-driving cars and cars with advanced technology, TSR technology
has become an essential real-world application. Our solution will be to classify traffic lights, stop signs,
speed limits, and crosswalks.

#### 1.2 Dataset
The original dataset is split into two folders: one that contains the png images, and another that
includes image annotations as xml files. There are 877 images with four distinct classes for the objective
of road sign detection. The classes are Traffic light, Stop signs, Speed limits and Crosswalks. After
reading information from both folders, the dataset is converted into a dataframe. The data frame consists
of 877 rows, a row for each image and 7 descriptive features and one class feature. The columns are:
filename, width, height, class label, xmin, ymin, xmax, and ymax. The filename attribute includes the
actual image. Class feature shows the class that image belongs to. Width and Height represent image
measurements, while the rest of the features display the bounding box coordinates.

#### 1.3 Proposed Analytics Solution
The objective of this project is to process a set of road sign images using GoogLeNet, a
convolutional neural network which is 22 layers deep (CNN). The aim is to classify the images into four
classes: traffic light, stop sign, speed limit, or crosswalk. Once the models are trained, they will be
evaluated, and recommendations for deployment will be provided. By leveraging CNNs, this project seeks
to achieve efficient and accurate classification of road signs.
For professional implementation of this project, several management tools are utilized. Trello is a
visual collaboration tool that gives teams a shared perspective on any project. It provides us a shared
space to organize and collaborate. We are using Trello to set project goals, guidelines, and deadlines.
Additionally, we are using Slack to communicate, share information, and ask questions and clarifications
about project details. Besides, we are using ML flow, which is a platform that helps manage end-to-end
machine learning lifecycle. It provides a suite of tools to track experiments, package code into
reproducible runs, and share and deploy models.Additionally, we are using ARCTIC, Georgia State’s
advanced computing technology. With the use of ARCTIC we are able to satisfy our high computational
needs in regards to running our models efficiently.

#### 1.4 Data Exploration and Preprocessing
Firstly, we visualized our images and annotations into a dataframe, then we converted our
dataframe to a CSV file to continue our exploratory data analysis (EDA). During our analysis we used
“value_counts'' to return the number of images in each class. The results indicated that the speed limit
class showed up for 75% of the data, with the remaining percentage evenly shared amongst the other
classes.

### 2 Data Visualization
###### Figure 1. Bar Plot of Classes
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/e0fda545-7b49-4a8a-9cd0-8918d31f8639)

Bar plot visualization for the four classes in the dataset

#### 2.1 Bar Plot of Classes

We used a bar chart to visualize the number of instances in each class. The bar chart reflected
the findings of value_counts. It is evident that the speed limit label is dominating while the other labels are
underrepresented. This calls for data augmentation to balance classes before applying classification
machine learning models to avoid model bias.

#### 2.1.1 Average Images Per Class
The average images represent the "typical" image in the dataset. It shows what the objects in the
images have in common, such as color, texture, and shape. Any variations among the images in the
dataset will be represented by the deviations from the average image. In this case, the average image is
mostly bluish gray, which does not reflect any specific part of the actual class. Additionally, the quality of
the image shows that input images are not very clear.
In the speed limit average image, the image is mostly bluish gray, which does not reflect any
specific part of the actual class. Additionally, the quality of the image shows that input images are not very
clear. This is due to the presence of several images in various magnifications, especially the presence of
far images of classes where the surroundings are covering most of the image rather than the sign itself.
This requires image cropping to create more focus on the traffic sign.
###### Figure 2. Stop Sign Average Image
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/c1bf0671-679e-4118-b0ed-0c6449ec5f4e)

A visualization of the average stop sign

###### Figure 3. Speed Limit Average Image
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/70afe6db-b859-4d29-a532-fc88a2324b2b)

A visualization of the speed limit average image
In the ‘Stop’ sign average image, the concentration of red in the middle and the presence of the
hexagon shape of the stop sign is evident. This shows that most ‘Stop’ sign images are close up images
where the sign is mostly in the center of the image. This can be useful in the classification process, since
the model will depend on those colors and shapes to learn during the training step.
###### Figure 4. Crosswalk Average Image
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/df15b38d-7973-4b9e-a5f6-08fb5a39f439)

A visualization of the crosswalk average image
The ‘Crosswalk’ average image has blue concentration in the middle with some visibility of the
triangular shape of the sign. This also reflects that most ‘Crosswalk’ images are focused on the traffic sign
without having a lot of empty space in the images.

###### Figure 5. Traffic Light Average Image
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/f543e7ef-255b-4014-8f50-37044dac50dd)

A visualization of the traffic light average image
The Traffic light average image also shows some aspects of a traffic light. Two circles are visible
and the variation between red and yellow colors are visible.

#### 2.1.2 Image Histograms

###### Figure 6. Image Color Histograms
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/2dc93043-957d-4e1e-a62e-0eaf22c123a2)

![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/00207ed0-8f91-4409-b222-8c90ad4e2f9c)

Image Color Histograms per class showing color distribution

We also used colored histograms to visualize the distribution of pixel values across the color
channels (R, G, B) for each class of images in the dataset.The x-axis of each histogram represents the
pixel value range, which is divided into a fixed number of bins; 256 in this case. The y-axis represents the
frequency of pixels in each bin, normalized by the total number of pixels in the image.
By examining the histograms, we can get a sense of the color composition of each class of
images. For example, if a class of images has a large peak in the red channel and a smaller peak in the
blue channel, we can infer that the images in that class tend to have a warm color tone with more red
than blue. If a class of images has roughly equal frequency across all color channels, we can infer that
the images in that class have a balanced color composition. If a class of images has very low frequency in
one or more color channels, we can infer that the images in that class are dominated by certain colors
and lack diversity in color composition.
For the ‘Crosswalk’ class, the colors have relatively equal frequency throughout the chart, so it
can be inferred that the color distribution in the class is balanced. There is a slightly higher distribution of
blue in most of these images, but overall it is relatively evenly balanced. For the ‘Speed Limit’ class, there
is slightly more distribution of green and blue colors than the red colors, but overall the color distribution is
balanced. For the ‘Stop Sign’ class, there is a large amount of variance for the amount of blue coloring in
the image. Some of the images are completely devoid of the blue color, while around 2.5% of the images
classified as stop signs are almost completely saturated with blue. For the ‘Traffic Light’ class, the color
distribution of the images are balanced. However, there are a few images completely devoid of one of the
colors.
The color variations in the ‘Crosswalk” and ‘Speedlimit” show that the model can learn from those
colors to classify between two classes. However, this is not the case for ‘Trafficlight’ and ‘Stop’, where the
color variations are similar following almost a unimodal distribution for colors. Besides, there are some
peaks in most of the classes around the sides of the images. These can be possible outliers that need to
be handled by cropping the images around those peaks. For instance, ‘Stop’ can be cropped between 25
and 225 to eliminate the peaks at both ends.

#### 2.2 Image Transformation and Data Augmentation
We cropped and rotated our images for model training. We have decided to crop and rotate our
images and save them into a new “updated_csv” file to test them in the future again. This technique can
help the model accurately classify the sign in an image even if the image has been modified or contains
some degree of distortion. We also calculated class weights to ensure that class balance is maintained,
so the model would not favor the dominating class over the underrepresented classes in the training
process. This approach is particularly important in situations where there is a significant class imbalance
in the training dataset.
###### Figure 7. Bounding Box on Crosswalk Sign Figure
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/3cde439c-f421-4714-be68-3a5b83780a8b)

###### 8. Bounding Box Around Rotated Crosswalk
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/285e1af3-ec3c-4211-936f-5010fc04dbe0)

Bounding box detecting sign based before rotation Bounding box detecting sign after rotation

#### 2.3 A Brief Description about CNN Model Selection
To classify our traffic signal images, we implemented a deep learning model in PyTorch. Deep
learning is a type of machine learning that uses neural networks with multiple layers to process and
analyze data. The neural networks are modeled after the human brain and can learn from large amounts
of data. Even though a neural network with only one layer can make approximate predictions, adding
more hidden layers can improve accuracy by refining and optimizing the learning process.
A Convolutional Neural Network (CNN) is a type of neural network commonly used in deep
learning for image and video recognition, analysis, and processing. CNNs consist of multiple
convolutional layers, which are responsible for identifying different features in an image or video, such as
edges, corners, and shapes. These layers are followed by one or more fully connected layers that help to
classify the image or video based on the features extracted by the convolutional layers. CNNs are highly
effective in image and video recognition tasks because they can automatically learn the relevant features
and patterns directly from the input data, without requiring manual feature extraction.
In this project, we used GoogLeNet to classify our traffic signal images. GoogLeNet is based on a
22 layers deep convolutional neural network architecture codenamed “Inception”, which was responsible
for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual
Recognition Challenge 2014 (ILSVRC 2014). It utilizes Inception modules, which allow the network to
choose between multiple convolutional filter sizes in each block. This module is an image model block
that aims to approximate an optimal local sparse structure in a CNN. Put simply, it allows for us to use
multiple types of filter size, instead of being restricted to a single filter size, in a single image block, which
we then concatenate and pass onto the next layer. In addition to the Inception module, GoogLeNet also
employs techniques such as max pooling, global average pooling, and batch normalization to improve the
performance and efficiency of the network. GoogLeNet is a highly influential architecture that has inspired
many subsequent neural network designs and has contributed significantly to the advancement of
computer vision research.

#### 2.4 CNN Model Specification
We utilized a pre-trained version of the network trained on either the ImageNet or Places365 data
sets. The network trained on ImageNet classifies images into 1000 object categories, such as keyboard,
mouse, pencil, and many animals. A pretrained CNN (Convolutional Neural Network) can help in
prediction by leveraging the knowledge learned from a large dataset to classify new, unseen images. By
using a pretrained CNN, we are taking advantage of the vast amount of data and computing resources
used to train the network, which can be difficult and expensive to replicate.
We split the updated dataset of the augmented images into two thirds, which is 0.67% of the
dataset as training set, and one third as testing set. We fed those images to the GoogLeNet model and it
classified the images into the four classes. We set the batch size to 32 which represents around 20% of
the training dataset, since the size of a batch must be less than or equal to the number of samples in the
training dataset. The model was tested several times with variable quantities of epochs to prevent
overfitting or underfitting. We set the learning rate to 0.001, which controls how quickly the nodes change
the values of their weights.
Our model worked best when we created a batch size of 32 and generated 100 epochs. We set
the specifications for the best results. Through our experimentation, we discovered that increasing the
batch size beyond a certain point would lead to overfitting, which is not desirable. Therefore, based on our
findings, we determined that a batch size of 32 was the optimal choice. Moreover, after careful
consideration, we selected the ADAM optimizer for our model. ADAM is an optimization algorithm that is
well-suited for training deep neural networks. It combines the benefits of two other optimization
techniques, namely AdaGrad and RMSProp, to provide a more robust and efficient optimizer. In addition,
ADAM has been shown to converge faster and to achieve better performance than other optimization
techniques in many cases. This is due in part to its ability to handle noisy gradients and to adapt to
changing gradients during training.

#### 2.5 Results
For evaluating the model, we used Cross Entropy Loss. Cross Entropy Loss measures the
dissimilarity between the predicted probabilities and the true labels. It computes the average difference
between the predicted probabilities and the true probabilities, where the true probability is 1 for the correct
class and 0 for all other classes. In other words, the Cross Entropy Loss penalizes the model for making
incorrect predictions and rewards it for making correct predictions. The value of Cross Entropy Loss
ranges from 0 to infinity, with 0 indicating perfect prediction accuracy and higher values indicating worse
performance. During training, the goal is to minimize the Cross Entropy Loss, which is achieved by
adjusting the model's parameters through backpropagation. Additionally, to ensure reliable performance
measurements, we evaluated the accuracy of our model across multiple experiments. This approach
allows us to obtain a more comprehensive understanding of the model's performance and to assess its
generalizability across different conditions. By repeating the evaluation multiple times and calculating the
average accuracy, we can obtain a more robust estimate of the model's performance and reduce the
influence of random fluctuations or outliers that may occur in a single experiment. Overall, this approach
enables us to make more informed decisions about the model's effectiveness and suitability for
deployment in real-world scenarios.
After running the neural network model with 5 epochs, the model had an accuracy of 87%. The
model was then run with 100 epochs, which had an accuracy of 92%. As shown in the confusion matrix
below, the model accurately classified images with speed limit signs, stop signs, and crosswalk signs,
with an accuracy of 87%, 89%, and 86% respectively. However, the model struggled to correctly detect
traffic lights demonstrated by the 64% accuracy. This may have been caused due to the relatively lower
amount of training data for traffic lights compared to the other classes. When increasing the batch size,
the model more accurately predicts the speed limit class at the expense of drastically decreasing the
accuracy of the other three classes.

###### Fig. 9A. Visualization of Confusion Matrix
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/389a87f9-a97d-48a5-9d6d-bd6858748b42)

Figure:Confusion matrix depicting 5 epochs and 32 batch size



######  Fig. 9B. Visualization of Confusion Matrix
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/12739675-d568-4d70-9b26-bd0f8ba77555)

Figure: Confusion matrix depicting 100 epochs and 32 batch size



###### Fig. 10. Visualization of Model Accuracy 
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/bbc47d9c-1a01-4caf-885e-74ad34c17472)

Figure: Testing and training accuracy trending together 



###### Fig. 11. Visualization of Model Loss
![image](https://github.com/Haninrefai/Deep-Learning-Based-Road-Sign-Detection/assets/89818668/f630f718-3f1b-4807-904b-02968d704b48)

Figure: Training Loss increasing with the increase of epochs


When analyzing the accuracy of our model, we observed that the training accuracy plateaus after
approximately 25 epochs, while the testing accuracy remains stable and fluctuates only slightly.
Meanwhile, our loss visualization revealed that the testing loss gradually increases as the training loss no
longer shows significant improvement after 20 epochs. These trends collectively suggest that our model
has reached its optimal performance and that further training iterations may not result in significant
improvements in accuracy.
While our model has demonstrated strong accuracy in identifying traffic signs, it is essential to
note that there may be factors that could impact its performance in real-world settings. For instance,
variations in lighting conditions, weather, and other environmental factors could affect the quality of the
images captured by autonomous vehicle cameras, potentially impacting the accuracy of our model's
predictions. As such, continued research and experimentation will be necessary to ensure that our model
remains robust and reliable in a range of different scenarios and conditions.

### 3 Possible Modifications and Improvements
While our current model has demonstrated a high level of accuracy in predicting sign classes, we
recognize that there is still significant potential for further improvement in our project. By continuing to
explore and experiment with cutting-edge techniques and algorithms in the field of computer vision, we
can potentially achieve even greater accuracy and robustness in our image classification system.
Additionally, ongoing research in areas such as transfer learning, data augmentation, and model
compression can provide valuable insights and tools for optimizing our model's performance and
efficiency. By staying at the forefront of these developments and continually seeking to enhance our
approach, we can ensure that our image classification system remains at the cutting edge of the field and
delivers maximum value to our users.
One potential enhancement to our approach is to utilize YOLO (You Only Look Once) algorithm to
generate precise bounding boxes around the objects of interest in the images. This technique can
improve the accuracy of object detection by providing a more fine-grained localization of the objects,
enabling more precise measurements of their size, position, and orientation. By incorporating YOLO into
our pipeline, we can further improve the performance and robustness of our image classification system,
making it more suitable for practical applications in domains such as autonomous driving, surveillance,
and robotics.
Another promising improvement to our image classification system is to incorporate the Faster
R-CNN (Region-based Convolutional Neural Network) algorithm. Faster R-CNN is a state-of-the-art object
detection model that can accurately identify and localize objects of interest in an image, while also
providing high detection speed. By integrating Faster R-CNN into our pipeline, we can potentially achieve
even higher accuracy and precision in object detection, which is crucial for many applications such as
self-driving cars and facial recognition systems. The use of a region-based CNN enables the model to
identify regions of interest in the image and apply a more detailed analysis, leading to a more fine-grained
and accurate classification. Ultimately, the addition of Faster R-CNN to our image classification system
could greatly enhance its effectiveness and suitability for a range of real-world applications.
An additional strategy that we could pursue to enhance our model's performance is to incorporate
American traffic signs into our dataset alongside the European signs already present. To achieve this, we
would need to assess the accuracy of our model in predicting American traffic signs before making any
adjustments to our parameters. Based on our findings, we could then modify our model and fine-tune its
parameters to ensure optimal performance in identifying both European and American traffic signs.
Integrating American traffic signs into our dataset would not only expand the scope and versatility
of our model but also increase its real-world applicability in regions where American traffic signs are
prevalent. This approach could potentially improve the safety and efficiency of autonomous vehicles on
American roads, further underscoring the importance of accurate traffic sign recognition technology.
Ultimately, by continuing to explore and experiment with innovative strategies for improving our model's
accuracy and robustness, we can help advance the state of the art in this critical area of research and
contribute to the ongoing development of safe and effective autonomous driving systems.

### 4 Conclusion
In summary, our study highlights the critical importance of accurate traffic sign recognition in the
context of autonomous vehicles. We found that the GoogLeNet model was able to achieve a high level of
accuracy in identifying various traffic signs, including speed limits, stop signs, and crosswalks, although it
faced challenges with traffic lights. Through our experimentation, we determined that the model worked
best with a batch size of 32 and trained for 20 epochs.
Future research regarding this project could focus on integrating state-of-the-art object detection
models such as YOLO or Faster R-CNN to enhance the precision and efficiency of traffic sign recognition.
Additionally, exploring novel techniques such as transfer learning and data augmentation could also yield
significant benefits in optimizing the performance of the model.
Overall, the successful implementation of accurate traffic sign recognition systems can have a
transformative impact on road safety, potentially preventing catastrophic accidents and saving countless
lives. As such, we believe that ongoing research and development in this area are essential to ensuring
the continued advancement of autonomous vehicle technology and improving the safety of our roads for
everyone.
