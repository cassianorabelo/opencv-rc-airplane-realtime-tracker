
# opencv-rc-airplane-realtime-tracker

![Computer vision RC plane tracking](https://i.imgur.com/dg7vvtK.gif)

The following code was developed as an exercise with the objective of getting acquainted with the OpenCV Computer Vision Library in C++.

The generated app is able to track, in realtime, RC planes in a provided footage taken with a hand held camera. Unfortunately, due to copyright restrictions, the original footage cannot be shared.

![Frame from the raw footage.](https://i.imgur.com/ypU1OZw.png)

The app is able to detect as soon as an rc plane enters the frame, its flight direction and the exact moment it crosses a vertical pole.

The algorightm is essentially composed of 3 blocks:
- Vertical pole detection;
- RC plane detection;
- Detection of the moment of crossing plane/pole.

The segmentation process is organized in multiple filtering steps with increasing computational costs, with each subsequent step applied to a smaller dataset which drastically improves the processing time.

The following images are taken from some of the steps required to successfully detect the RC plane.

![Some of the steps taken to detect the object.](https://i.imgur.com/lLAJUwd.jpg)

Raw footage showing a RC plane in frame (a), grayscale conversion (b), image binarization(c), contour detection(d), optical flow based feature detection(e), chromatic segmentation filtering(f,g) with its false color image (f) and generated result(g). Region of interest (ROI) for the detection of the vertical pole (h,i) and debug view after all the detection steps have taken place(j) where it is possible to see a RC plane and vertical pole with their respective boundaries highlighted.

![Detection result with debug mode on](https://i.imgur.com/3y6uCnj.jpg)
