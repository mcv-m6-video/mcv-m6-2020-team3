

## Master in Computer Vision Barcelona 2020 - Module 6: Video Analysis


## Abstract

### Week 1

* Task 1: Detection metrics.
* Task 2: Detection metrics. Temporal analysis.
* Task 3: Optical flow evaluation metrics.
* Task 4: Visual representation optical flow.


In this week we setup the different metrics used and get used to the dataset
we used in the following weeks. 

### Week 2
+ #### Background estimation
  + Model the background pixels of a video sequence using a simple statistical model to classify the background / foreground
    + Single Gaussian per pixel 
    + Adaptive / Non-adaptive 
  + The statistical model will be used to preliminary classify foreground
 + #### Comparison with more complex models 


#### Task 1: Gaussian distribution and Evaluate
+ One Gaussian function to model each background pixel
  + First 25% of the test sequence to model background
  + Mean and variance of pixels
+ Second 75% to segment the foreground and evaluate
1.2 & 1.3: Evaluate results

#### Task 2: Recursive Gaussian modeling. Evaluate and compare to non-recursive
+ Adaptive modelling
  + First 25% frames for training
  + Second 75% left background adapts
+ Best pair of values (𝛼, ⍴) to maximize mAP
  + Two methods:
    + Obtain first the best 𝛼 for non-recursive, and later estimate ⍴ for the recursive cases
    + Optimize (𝛼, ⍴) together with grid search or random search (discuss pros & cons).
#### Task 3: Compare with state-of-the-art and Evaluation
+ P. KaewTraKulPong et.al. An improved adaptive background mixture model for real-time tracking with shadow detection. In Video-Based Surveillance Systems, 2002. Implementation: BackgroundSubtractorMOG (OpenCV)
+ Z. Zivkovic et.al. Efficient adaptive density estimation per image pixel for the task of background subtraction, Pattern Recognition Letters, 2005. Implementation: BackgroundSubtractorMOG2 (OpenCV)
+ L. Guo, et.al. Background subtraction using local svd binary pattern. CVPRW, 2016. Implementation: BackgroundSubtractorLSBP (OpenCV)

#### Task 4: Color sequences
+ Update with support color sequences


### Week 3
#### T1 Object Detection
##### T1.1 Off-the-shelf
Use Mask R-CNN object detector and we run it to detect cars in each frame (separately). The dataset used to train your network contains the class ‘car’.
##### T1.2 Fine-tuning
Fine tuning from ourdata will in general require two steps:
  + Defining the new dataset 
  + Fine-tuning the last layer(s) from a pre-trained model 
  
  
We try different data partitions:
  + Strategy A: First 25% frames for training. Second 75% for test.
  + Strategy B: Random 25%
  + Strategies C & D: Instead of 25%, try with different % for training/test partitions.

#### T2 Object Tracking 
##### T2.1 Tracking by Overlap
##### T2.2 Tracking with a Kalman Filter
#### T2.3 (optional)  IDF1 for Multiple Object Tracking
#### T3 (optional) CVPR 2019 AI City Challenge 
### Week 4

### Week 5



# mcv-m6-2020-team3
Team 3
## Team members

|      Member     |           Email          |
|:---------------:|:------------------------:|
|  Yixiong Yang| yixiong.yang@e-campus.uab.cat |
|     Sanket Biswas   |    sanket.biswas@e-campus.uab.cat   |
|  Gabriela Cordova |    gabrielaelizabeth.cordova@e-campus.uab.cat    |
| Marc Oros Casanas  |marc.oros@e-campus.uab.cat |
| Keyao Li | keyao.li@e-campus.uab.cat|

