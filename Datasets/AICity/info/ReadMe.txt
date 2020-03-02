%******************************************************************************************************************%
% The AIC19 benchmark is captured by 40 cameras in real-world traffic surveillance environment.                    %
% A total of 666 vehicles are annotated in 5 different scenarios. 3 of the scenarios are used for training. The    %
% remaining 2 scenarios are for testing.                                                                           %
% There are 195.03 minutes of videos in total. The length of the training videos is 58.43 minutes, and the testing %
% videos 136.60 minutes.                                                                                           %
%******************************************************************************************************************%

Content in the directory:

1. "train/*". It contains all the subsets for training. 
2. "test/*". It contains all the subsets for testing. 
3. "train(test)/<subset>/<cam>/vdo.avi". They are the test videos. 
4. "train(test)/<subset>/<cam>/roi.jpg". They are the region of interest (ROI), where the white area covers the entire body of each vehicle object. 
5. "train/<subset>/<cam>/gt/gt.txt". They list the ground truths of MTMC tracking in the MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1]. Only vehicles that pass through at least 2 cameras are taken into account. 
6. "train(test)/<subset>/<cam>/det/det_*.txt". They are the detection results from different baselines in the MOTChallenge format [frame, -1, left, top, width, height, conf, -1, -1, -1]. The reference for each baseline method is given as follows.
[YOLOv3] Redmon, Joseph and Farhadi, Ali, "YOLOv3: An Incremental Improvement," arXiv, 2018.
[SSD] Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C., "SSD: Single Shot MultiBox Detector," ECCV, 2016.
[Mask/Faster R-CNN] He, Kaiming and Gkioxari, Georgia and Dollár, Piotr and Girshick, Ross, "Mask R-CNN," ICCV, 2017.
7. "train(test)/<subset>/<cam>/mtsc/mtsc_*.txt". They are the MTSC tracking results from different baselines in the MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1]. The reference for each baseline method is given as follows.
[Deep SORT] Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich, "Simple Online and Realtime Tracking with a Deep Association Metric," ICIP, 2017.
[Tracklet Clustering] Tang, Zheng and Wang, Gaoang and Xiao, Hao and Zheng, Aotian and Hwang, Jenq-Neng, "Single-camera and Inter-camera Vehicle Tracking and 3D Speed Estimation Based on Fusion of Visual and Semantic Features," CVPRW, 2018.
[MOANA] Tang, Zheng and Hwang, Jenq-Neng, "MOANA: An Online Learned Adaptive Appearance Model for Robust Multiple Object Tracking in 3D," arXiv, 2019.
8. "train(test)/<subset>/<cam>/segm/segm_mask_rcnn.txt". They are the segmentation results from Mask R-CNN (each line corresponds to the detection results in det_mask_rcnn.txt). 
9. "train(test)/<subset>/<cam>/calibration.txt". They are the manual calibration results. The first line gives the 3x3 homography matrix projecting from the GPS coordinates to the 2D image pixel location. For fish-eye cameras, the second line gives the distortion coefficients. 
10. "list_cam.txt". It lists the subfolder of each video for training/testing.
11. "cam_loc/<subset>.png". They are the maps with the camera locations. Since we do not have access to the exact GPS location of each camera, the GPS location for the approximate center of each scenario is provided. 
The GPS location for S01.png is 42.525678, -90.723601.
The GPS location for S02.png is 42.491916, -90.723723.
The GPS location for S0345.png is 42.498780, -90.686393.
12. "cam_timestamp/<subset>.txt". They list the (starting) timestamps of videos in seconds for each of the 5 scenarios. Note that due to noise in video transmission, which is common in real deployed systems, some frames are skipped within some videos, so they are not perfectly aligned. The frame rates of all the videos are 10 FPS, except for c015 in S03, whose frame rate is 8 FPS. 
13. "cam_framenum/<subset>.txt". They list the numbers of video frames for each of the 5 scenarios. 
14. "amilan-motchallenge-devkit/". This is an extension of the Matlab evaluation code for MOTChallenge (https://bitbucket.org/amilan/motchallenge-devkit/). When running demo_evalAIC19.m, an example for the evaluation of the training set will be automatically processed. 
15. "DataLicenseAgreement_AICityChallenge.pdf". The license agreement for the usage of this dataset. 

If you have any question, please contact aicitychallenge2019@gmail.com.
