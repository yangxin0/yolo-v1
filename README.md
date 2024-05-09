<!--


mlpi
title: You Only Look Once: Unified, Real-Time Object Detection (YOLOv1)
category: Architectures/Convolutional Neural Networks
images: results/many_people_horse.png, results/multiple_cars.png, results/person_motorbike.png, results/multiple_people.png, results/many_people_motorbike.png, results/many_people.png
-->

<h1 align="center">YOLOv1</h1>
           
PyTorch implementation of the YOLO architecture presented in "You Only Look Once: Unified, Real-Time Object Detection" by Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi


## Methods
For the sake of convenience, PyTorch's pretrained `ResNet50` architecture was used as the backbone for the model instead of `Darknet`. However, the detection
layers at the end of the model exactly follow those described in the paper. The data was augmented by randomly scaling dimensions, 
shifting position, and adjusting hue/saturation values by up to 20% of their original values.


## References
[[1](https://arxiv.org/abs/1506.02640)] Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. _You Only Look Once: Unified, Real-Time Object Detection_. arXiv:1506.02640v5 [cs.CV] 9 May 2016

[[2](https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2)] Towards Datascience. _mAP (mean Average Precision) might confuse you!_

<!-- 
[[2](https://arxiv.org/abs/1612.08242)] Joseph Redmon, Ali Farhadi. _YOLO9000: Better, Faster, Stronger_. arXiv:1612.08242v1 [cs.CV] 25 Dec 2016

[[3](https://arxiv.org/abs/1804.02767v1)] Joseph Redmon, Ali Farhadi. _YOLOv3: An Incremental Improvement_. arXiv:1804.02767v1 [cs.CV] 8 Apr 2018
 -->
