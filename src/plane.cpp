//
//  plane.cpp
//  DAI-questao-03-avião
//
//  Created by Cassiano Rabelo on oct/16.
//  Copyright © 2016 Cassiano Rabelo. All rights reserved.
//

#include "plane.hpp"

using namespace std;
using namespace cv;

bool gDebug = true;
bool gPause = false;

void onSkipFrames(VideoCapture &cap, int numFrames) {
  int newFrame = 0;
  int curFrame = (int)cap.get(CV_CAP_PROP_POS_FRAMES);
  
  if (numFrames > 0) {
    if (curFrame <= 370) {
      newFrame = 390;
    } else if (curFrame > 370 && curFrame < 690) {
      newFrame = 690;
    } else if (curFrame > 690 && curFrame < 950) {
      newFrame = 950;
    } else if (curFrame > 950 && curFrame < 1255) {
      newFrame = 1255;
    } else if (curFrame > 1200 && curFrame < 1550) {
      newFrame = 1550;
    } else if (curFrame >= 1550) {
      newFrame = 1830;
    }
    
  } else {
    if (curFrame <= 370) {
      newFrame = 360;
    } else if (curFrame > 370 && curFrame < 690) {
      newFrame = 380;
    } else if (curFrame > 690 && curFrame < 950) {
      newFrame = 680;
    } else if (curFrame > 950 && curFrame < 1255) {
      newFrame = 940;
    } else if (curFrame > 1250 && curFrame < 1550) {
      newFrame = 1180;
    } else if (curFrame > 1520 && curFrame < 1830) {
      newFrame = 1510;
    } else if (curFrame >= 1830) {
      newFrame = 1830;
    }
  }
  
  cap.set( CAP_PROP_POS_FRAMES, newFrame );
}

void isolatePole(Mat &frame,
                 UMat &frameGray,
                 UMat &frameSegmentedPole,
                 const Rect &roiRect,
                 vector<vector<Point>> &poleContours,
                 Point &poleCenterMass
                 ) {
  
  segment(frameGray, frameSegmentedPole);
  Mat roi(frameSegmentedPole.getMat(ACCESS_WRITE), roiRect);
  
  // FIND POLE CONTOURS
  Mat contoursImg;
  roi.copyTo(contoursImg);
  
  imwrite("04_pole_roi.png", contoursImg);
  
  detectPole(contoursImg, poleContours);
  
  double angle; // stores pole angle
  for(unsigned int i = 0; i < poleContours.size(); i++) {
    angle = getOrientation(poleContours[i], frame, poleCenterMass, roiRect.tl());
  }

}

void matPrint(Mat &img, Point pos, Scalar fontColor, const string &ss) {
  int fontFace = FONT_HERSHEY_DUPLEX;
  double fontScale = 0.5;
  int fontThickness = 1;
  Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);
  
  Point org;
  org.x = pos.x - fontSize.width;
  org.y = pos.y - fontSize.height/ 2;
  putText(img, ss, org, fontFace, fontScale, Scalar(255, 255, 255), fontThickness, LINE_AA);
}

void convertToGrey(InputArray _in, OutputArray _out) {
  _out.create(_in.getMat().size(), CV_8UC1);
  if(_in.getMat().type() == CV_8UC3)
    cvtColor(_in.getMat(), _out.getMat(), COLOR_BGR2GRAY);
  else
    _in.getMat().copyTo(_out);
}

void pause(VideoCapture &cap) {
  gPause = !gPause;
  int curFrame = cap.get(CAP_PROP_POS_FRAMES);
  cout << "PAUSE = " << (gPause?"true":"false") << " - FRAME: " << curFrame << endl;
}

/**
 * @brief Threshold input image
 */
void segment(InputArray _in, OutputArray _out) {
  // THRESHOLD
  threshold(_in, _out, 125, 255, THRESH_BINARY | THRESH_OTSU);
  
  // ERODE / DILATE
  int morph_size = 1;
  Mat elErode = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size+1, 2*morph_size+1 ) );
  erode(_out, _out, elErode, Point(-1, -1), 8, BORDER_DEFAULT);
  
  Mat elDilate = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size+1, 2*morph_size+1 ) );
  dilate(_out, _out, elDilate, Point(-1, -1), 4, BORDER_DEFAULT);
  
  // INVERT
  bitwise_not(_out, _out);
}

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
                      const Point offset) {
  
  //Construct a buffer used by the pca analysis
  Mat data_pts = Mat((uint)pts.size(), 2, CV_64FC1);
  for (int i = 0; i < data_pts.rows; ++i) {
    data_pts.at<double>(i, 0) = pts[i].x;
    data_pts.at<double>(i, 1) = pts[i].y;
  }
  
  //Perform PCA analysis
  PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
  
  //Store the position of the center of mass
  posCenter = Point(pca_analysis.mean.at<double>(0, 0) + offset.x,
                    pca_analysis.mean.at<double>(0, 1) + offset.y);
  
  //Store the eigenvalues and eigenvectors
  vector<Point2d> eigen_vecs(2);
  vector<double> eigen_val(2);
  for (int i = 0; i < 2; ++i) {
    eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                            pca_analysis.eigenvectors.at<double>(i, 1));
    
    eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
  }
  
  // Draw the principal components
  if (gDebug) {
    Point start(posCenter - 0.5 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]));
    Point end(posCenter + 0.5 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]));
    
    line(img, start, end, CV_RGB(255, 255, 0), 2, LINE_AA);
    line(img, posCenter, posCenter + 0.02 * Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]) , CV_RGB(0, 255, 255), 2, LINE_AA);
    circle(img, posCenter, 5, Scalar(255, 0, 0), CV_FILLED, LINE_AA);
  }
  
  return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}

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
               ) {
  
  
  Mat contoursImg;
  _in.getMat().copyTo(contoursImg);
  double aspectRatio = contoursImg.cols/(float)contoursImg.rows;
  
  int minPerimeterPoints = 50;
  int maxPerimeterPoints = 3000;
  
  if (aspectRatio < 1) // image is taller than wider (needed due to bug in reading vertical videos)
  {
    maxPerimeterPoints = 500;
  }
  
  vector<vector<Point>> tmpCandidates;
  int maxMagnitude = 20;
  
  Mat temp;
  contoursImg.copyTo(temp);
  cvtColor(temp, temp, CV_GRAY2BGR);
  
  vector< vector< Point > > _contours;
  findContours(contoursImg, _contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
  
  drawContours(temp, _contours, -1, Scalar(0, 0, 255), 2, LINE_AA);
  imwrite("05.png", temp);
  
  for(unsigned int i = 0; i < _contours.size(); i++) {
    
    // check amount of perimeter points
    if(_contours[i].size() < minPerimeterPoints || _contours[i].size() > maxPerimeterPoints) continue;
    
    Rect bounding = boundingRect(_contours[i]);
    
    // check if the aspect ratio is valid - something is too long...
    float aspectRatio = (float)bounding.width / bounding.height;
    if (aspectRatio > 5) continue;
    
    // check position in frame - ignore if too low
    if (bounding.y > _in.rows()*.8) continue;
    
    // check area
    double area = contourArea(_contours[i]);
    if (area < 800) continue;
    
    // check to see if most of the contour is composed of blue color
    Mat roi(_inColor.getMat(), bounding);
    Mat roiHSV, inRangeRes;
    roi.copyTo(roiHSV);
    cvtColor(roi, roiHSV, CV_BGR2HSV);
    inRange(roiHSV, Scalar(70, 50, 50), Scalar(130, 255, 255), inRangeRes );
    Scalar sumResult = sum(inRangeRes) / (inRangeRes.cols*inRangeRes.rows);
    
    if (sumResult[0] > 250) {
      // cout << "sum: " << sumResult << endl;
      continue;
    }
            imshow("roi", roiHSV);
            imshow("sky", inRangeRes);
    
    // if convex, slim chance of being a plane
    if(isContourConvex(_contours[i])) continue;
    
    tmpCandidates.push_back(_contours[i]);
  }
  
  if (tmpCandidates.size() == 0) {
    return;
  } else if (tmpCandidates.size() == 1) {
    candidates.push_back(tmpCandidates[0]);
    return;
  }
  
  // test to see if the points are inside a contour
  vector<double> magnitudeMeanValue(tmpCandidates.size(), 0.0);
  for (size_t i = 0; i < tmpCandidates.size(); ++i) {
    int numPointsInsideContour = 0;
    for (size_t j = 0; j < nextPts.size(); ++j) {
      if (pointPolygonTest(tmpCandidates[i], nextPts[j], false) >= 0) {
        magnitudeMeanValue[i] += magnitude[j];
        numPointsInsideContour++;
      }
    }
    
    // magnitudeMeanValue[i] = numPointsInsideContour?magnitudeMeanValue[i]/(float)numPointsInsideContour:0;
    magnitudeMeanValue[i] = numPointsInsideContour?magnitudeMeanValue[i]:0;
  }
  
  double maxVal = -1;
  double maxValPos = 0;
  for (size_t i = 0; i < tmpCandidates.size(); ++i) {
    if (magnitudeMeanValue[i] > maxVal) {
      maxVal = magnitudeMeanValue[i];
      maxValPos = i;
    }
  }
  
  if (maxVal < maxMagnitude) {
    return;
  }
  
  // cout << "mag: " << magnitudeMeanValue[maxValPos] << endl;
  candidates.push_back(tmpCandidates[maxValPos]);
  
  contoursImg.copyTo(temp);
  cvtColor(temp, temp, CV_GRAY2BGR);
  drawContours(temp, _contours, -1, Scalar(0, 0, 255), 2, LINE_AA);
  imwrite("06.png", temp);
  
}

bool opticalFlow(InputOutputArray &flowPrev,
                 InputArray frameGray,
                 vector<Point2f> &pts,
                 vector<Point2f> &nextPts,
                 vector<unsigned char> &status,
                 vector<float> &err
                 ) {
  
  const int numPoints = 500;
  const int minDist = 0;
  
  pts.clear();
  goodFeaturesToTrack(frameGray, pts, numPoints, .01, minDist);
  
  
  Mat temp;
  frameGray.copyTo(temp);
  cvtColor(temp, temp, CV_GRAY2BGR);
  
  for (uint i=0;i<pts.size();i++) {
    circle(temp, Point(pts[i].x, pts[i].y), 5, Scalar(0, 255, 255), CV_FILLED, LINE_AA);
  }
  imwrite("06.png", temp);
  
  
  if(pts.size() == 0) {
    frameGray.copyTo(flowPrev);
    return false;
  }
  
  
  
  calcOpticalFlowPyrLK(flowPrev, frameGray, pts, nextPts, status, err);
  return true;
}

void drawCrossingSign(InputArray _in,
                      const Point &pt1,
                      const Point &pt2)
{
  line(_in.getMat(), pt1, pt2, Scalar(0,0,255), 50, LINE_AA);
}

/**
 * @brief
 * _in: grayscale frame
 * _out: segmented frame
 */
void segmentUAV(InputArray _in, OutputArray _out) {
  
  threshold(_in, _out, 240, 255, THRESH_BINARY | THRESH_OTSU);
  
  // ERODE
  int morph_size = 1;
  Mat elErode = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ) );
  erode(_out, _out, elErode, Point(-1, -1), 10, BORDER_DEFAULT);
  
  // DILATE
  Mat elDilate = getStructuringElement( MORPH_RECT, Size( 2*morph_size+1, 2*morph_size+1 ) );
  dilate(_out, _out, elDilate, Point(-1, -1), 4, BORDER_DEFAULT);
  
  // INVERT
  bitwise_not(_out, _out);
}

/**
 * @brief Detect the flag pole
 */
void detectPole(InputArray _in,
                vector< vector< Point > > &_candidates) {
  
  int minPerimeterPoints = 450;
  int maxPerimeterPoints = 600;
  
  Mat contoursImg;
  _in.getMat().copyTo(contoursImg);
  
  vector< vector< Point > > _contours;
  findContours(contoursImg, _contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
  
  Mat _inTmp;
  _in.copyTo(_inTmp);
  cvtColor(_inTmp, _inTmp, CV_GRAY2BGR);
  
  drawContours(_inTmp, _contours, -1, Scalar(0, 0, 255), 2, LINE_AA);
  imwrite("04_pole_contours.png", _inTmp);
  
  
  Mat displayOrientation;
  displayOrientation.create( _in.size(), CV_8UC3 );
  
  for(unsigned int i = 0; i < _contours.size(); i++) {
    
    // check if image goes from top to bottom
    Rect bounding = boundingRect(_contours[i]);
    if (bounding.y > 1 || bounding.height < (_in.getMat().rows-2) ) continue;  // the -2 is to fit the contour inside the frame
    
    // check if the aspect ratio is valid
    float aspectRatio = (float)bounding.width / bounding.height;
    if (aspectRatio > 0.2) continue;
    
    // check perimeter
    if(_contours[i].size() < minPerimeterPoints || _contours[i].size() > maxPerimeterPoints) continue;
    
    // check is square and is convex
    double arcLen = arcLength(_contours[i],true);
    vector< Point > approxCurve;
    approxPolyDP(_contours[i], approxCurve, arcLen * 0.005, true);
    if(approxCurve.size() < 4 || approxCurve.size() > 6 || !isContourConvex(approxCurve)) continue;
    _candidates.push_back(_contours[i]);
  }
}

void calcOFlowMagnitude(const vector<Point2f> &prevPts,
                        const vector<Point2f> &nextPts,
                        const vector<uchar> &status,
                        vector<double> &magnitude) {
  
  magnitude.clear();
  magnitude = vector<double>(status.size(), 0);
  
  for (size_t i = 0; i < prevPts.size(); ++i) {
    if (status[i]) {
      Point2f p = prevPts[i];
      Point2f q = nextPts[i];
      double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );
      magnitude[i] = hypotenuse;
    }
  }
}