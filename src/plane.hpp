//
//  plane.hpp
//  DAI-questao-03-avião
//
//  Created by Cassiano Rabelo on oct/16.
//  Copyright © 2016 Cassiano Rabelo. All rights reserved.
//

#ifndef plane_hpp
#define plane_hpp

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>

using namespace std;
using namespace cv;

extern bool gDebug; // debug mode ON/OFF
extern bool gPause; // is video paused? ON/OFF

// Detection of the pole
void isolatePole(Mat &frame, UMat &frameGray, UMat &frameSegmentedPole, const Rect &roiRect, vector<vector<Point>> &poleContours, Point &poleCenterMass);

// Threshold input image
void segment(InputArray _in, OutputArray _out);

// Get orientation of the contour using PCA
// Reference: https://goo.gl/IWFUcP
double getOrientation(vector<Point> &pts,
                      Mat &img,
                      Point &posCenter,
                      const Point offset = Point(0,0));

//Detects the UAV based on contour conditions
void detectUAV(InputArray _in,
               InputArray _inColor,
               vector<vector<Point>> &candidates,
               const vector<Point2f> &prevPts,
               const vector<Point2f> &nextPts,
               const vector<uchar> &status,
               vector<double> &magnitude
               );

bool opticalFlow(InputOutputArray &flowPrev,
                 InputArray frameGray,
                 vector<Point2f> &pts,
                 vector<Point2f> &nextPts,
                 vector<unsigned char> &status,
                 vector<float> &err
                 );

// Signals on screen that the plane has crossed the pole
void drawCrossingSign(InputArray _in,
                      const Point &pt1,
                      const Point &pt2);

// segments the image with UAV parameters
void segmentUAV(InputArray _in, OutputArray _out);

// Detect the flag pole
void detectPole(InputArray _in, vector< vector< Point > > &_candidates);

void calcOFlowMagnitude(const vector<Point2f> &prevPts,
                        const vector<Point2f> &nextPts,
                        const vector<uchar> &status,
                        vector<double> &magnitude);

// Jumps the # of frames provided
void onSkipFrames(VideoCapture &cap, int numFrames);

// Color conversion
void convertToGrey(InputArray _in, OutputArray _out);

// Pauses the video
void pause(VideoCapture &cap);

// Adds textual info to the provided image
void display(Mat &img, Point pos, Scalar fontColor, const string &ss);


#endif /* plane_hpp */
