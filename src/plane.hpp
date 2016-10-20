//
//  plane.hpp
//  DAI-questao-03-avião
//
//  Created by Cassiano Rabelo on 10/20/16.
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

extern bool gDebug;
extern bool gPause;

void onSkipFrames(VideoCapture &cap, int numFrames, bool jumpDirectly = false);
void matPrint(Mat &img, Point pos, Scalar fontColor, const string &ss);
void convertToGrey(InputArray _in, OutputArray _out);
void pause(VideoCapture &cap);

/**
 * @brief Threshold input image
 */
void segment(InputArray _in, OutputArray _out);

/**
 * @brief Get orientation of the contour using PCA
 * Reference: https://goo.gl/IWFUcP
 * PARAMS:
 * pts: points from findContours
 * img: image to draw the results on
 * offset: amount to displace the center found
 * posCenter: stores the center of mass found
 */
double getOrientation(vector<Point> &pts,
                      Mat &img,
                      Point &posCenter,
                      const Point offset = Point(0,0));

/**
 * @brief Detects the UAV based on contour conditions
 */
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

void drawCrossingSign(InputArray _in,
                      const Point &pt1,
                      const Point &pt2);

/**
 * @brief
 * _in: grayscale frame
 * _out: segmented frame
 */
void segmentUAV(InputArray _in, OutputArray _out);

/**
 * @brief Detect the flag pole
 */
void detectPole(InputArray _in,
                vector< vector< Point > > &_candidates);

void calcOFlowMagnitude(const vector<Point2f> &prevPts,
                        const vector<Point2f> &nextPts,
                        const vector<uchar> &status,
                        vector<double> &magnitude);

void drawArrows(UMat& _frame,
                const vector<Point2f>&prevPts,
                const vector<Point2f>&nextPts,
                const vector<uchar>&status,
                Scalar line_color = Scalar(0, 0, 255));


#endif /* plane_hpp */
