

## Master in Computer Vision Barcelona 2020 - Module 6: Video Analysis


## Abstract
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

