//
//  main.cpp
//  DAI-questao-03-avião
//
//  Created by Cassiano Rabelo on oct/16.
//  Copyright © 2016 Cassiano Rabelo. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>

#include "plane.hpp"

using namespace std;
using namespace cv;

void help(char exe[])
{
    cout
    << "------------------------------------------------------------------------------" << endl
    << "DAI . DETECAO E ANALISE DE IMAGENS"                                             << endl
    << "2016/2o. SEMESTRE"                                                              << endl
    << "ALUNO: CASSIANO RABELO E SILVA"                                                 << endl
    << "QUESTAO #3 . AEROMODELOS"                                               << endl << endl
    << "Utilizacao:"                                                                    << endl
    << exe << " --video <video> [--output <video>]"                                     << endl
    << "------------------------------------------------------------------------------" << endl
    << "Utilizando OpenCV " << CV_VERSION << endl << endl;
}

/////////////////////////////////////

int main(int argc, char *argv[]) {
    
    if (argc == 1) {
        help(argv[0]);
        return -1;
    }
    
    string output;                      // recorded video
    string input;                       // input video
    bool writeOutput = false;           // write output?
    VideoWriter outputVideo;
    
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--video") {
            input = argv[++i];
        } else if (string(argv[i]) == "--output") {
            output = argv[++i];
            writeOutput = true;
        } else if (string(argv[i]) == "--help") {
            help(argv[0]);
            return -1;
        } else {
            cout << "Parametro desconhecido: " << argv[i] << endl;
            return -1;
        }
    }
    
    // MATs
    Mat frame;
    UMat frameGray, frameSegmentedPole, frameSegmentedPlane;
    
    // opticalFlow
    UMat flowPrev;
    uint points = 1000;
    vector<Point2f> pts(points);
    vector<Point2f> nextPts(points);
    vector<unsigned char> status(points);
    vector<float> err;
    
    VideoCapture inputVideo;
    inputVideo.open(input);
    CV_Assert(inputVideo.isOpened());
    
    // output resolution based on input
    Size S = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),
                  (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));
    cout << S.width << " - " << S.height << endl;
    
    // ROI top left (TL) and bottom right (BR) in respect to the full frame
    Point roiTL = Point(S.width*0.3, S.height*0.35);
    Point roiBR = Point(S.width*0.7, S.height*0.7);

    // get frame
    inputVideo >> frame;
    if (frame.empty()) {
        cerr << "erro" << endl;
        return -1;
    }
    convertToGrey(frame, flowPrev);
    
    Point planePosPrev(0,0);        // plane position: previous
    Point planePosCurr(0,0);        // plane position: current
    bool planeVisible = false;      // plane in frame?
    int planeDirection = 0;         // plane moving direction
    uint frameLastSeen = 0;         // plane last seen frame
    uint frameIntervalBeforeReset = 3;          // min # of frames without plane to reset the system
    uint maxPlaneDisplacement = S.width * 0.8;  // maximum distance plane will travel between frames
    const uint marginSize = S.width * 0.15;     // when the plane appears, it must be inside the margin (in pixels)
    
    // Output
    if (writeOutput) {
        int FPS = 30;
        outputVideo = VideoWriter(output, CV_FOURCC('M','J','P','G'), FPS, S, true);
        
        if (!outputVideo.isOpened())
            cerr << "Nao foi possivel abrir o arquivo de video para escrita" << endl;
        
        cout
        << "Salvando frames no arquivo: " << output << " com as seguintes caracteristicas:" << endl
        << "Largura=" << S.width << endl
        << "Altura=" << S.height << endl
        << "FPS=" << FPS << endl
        << "CODEC: MJPG - Motion JPEG" << endl
        << "------------------------------------------------------------------------------" << endl
        << "PARA SAIR APERTE A TECLA 'ESC'" << endl;
        
    }
    
    // LOOP
    for (;;) {
        
        char key = (char)waitKey(10); // 10ms/frame
        if(key == 27) break;
        
        // allows to skip frames (faster to debug)
        switch (key)
        {
            case '[':
                onSkipFrames(inputVideo, -1);
                break;
            case ']':
                onSkipFrames(inputVideo, +1);
                break;
            case 'p':
                pause(inputVideo);
                break;
        }
        
        if (gPause)
            continue;
        
        inputVideo >> frame;
        
        if (frame.empty()) {
            break;
        }
        
        uint curFrame = inputVideo.get(CAP_PROP_POS_FRAMES);
        
        convertToGrey(frame, frameGray);
        segment(frameGray, frameSegmentedPole);
        
        // ISOLATE THE FLAG POLE
        Rect roiRect(roiTL, roiBR);
        Mat roi(frameSegmentedPole.getMat(ACCESS_WRITE), roiRect);
        
        // FIND POLE CONTOURS
        
        Mat contoursImg;
        roi.copyTo(contoursImg);
        
        vector<vector<Point>> poleContours;
        detectPole(contoursImg, poleContours);
        
        //Store the position of the object
        Point poleCenterMass; // stores pole center of mass
        double angle; // stores pole angle
        for(unsigned int i = 0; i < poleContours.size(); i++) {
            angle = getOrientation(poleContours[i], frame, poleCenterMass, roiTL);
        }
        
        // DRAW POLE
        //    drawContours(frame, poleContours, -1, Scalar(0,0,255), 1, LINE_AA, noArray(), INT_MAX, roiTL);
        //    imshow("Detected candidates", frame);
        
        // DETECT AIRPLANE
        vector< vector< Point > > planeContours;
        segmentUAV(frameGray, frameSegmentedPlane);
        //  imshow("segmented UAV", frameSegmentedPlane);
        
        // calculate, using optical flow, the magnitude of the movement
        if (!opticalFlow(flowPrev, frameGray, pts, nextPts, status, err)) continue;
        vector<double> magnitude;
        calcOFlowMagnitude(pts, nextPts, status, magnitude);
        detectUAV(frameSegmentedPlane, frame, planeContours, pts, nextPts, status, magnitude);
        
         Mat mask(frameSegmentedPlane.rows, frameSegmentedPlane.cols, CV_8UC1, Scalar(0));
         drawContours(mask, planeContours, -1, Scalar(255), CV_FILLED);
         imshow("mask", mask);
        
        
        // _CR_ DEBUG
        //        if (planeContours.size() > 1) {
        //            cout << "[" << curFrame << "] " << "[num. contours: " << planeContours.size() << "]" << endl;
        //        }
        
        Rect bounding;
        
        // tem q resolver isso aqui pra quando for > 1
        if (planeContours.size() == 1)
        {
            frameLastSeen = curFrame;
            bounding = boundingRect(planeContours[0]);
            
            if (!planeVisible)
            {
                bool planeEnteringFrame = !(bounding.x > marginSize && bounding.x < (S.width - marginSize));
                if (planeEnteringFrame) // is the plane entering the frame or it "teleported"?
                {
                    planeVisible = true; // plane just appeared;
                }
            }
            else
            { // planeVisible
                planePosPrev = planePosCurr;
                planePosCurr = Point(bounding.x, bounding.y);
            }
        }
        else // contours != 1
        {
            if (planeVisible && curFrame - frameLastSeen > frameIntervalBeforeReset)
            { // the plane was visible and a few frames have ellapsed with no plane in sight...
                planeVisible = false;
                planeDirection = 0; // no direction on record
            }
        }
        
        if (planeVisible)
        {
            string movingTxt;
            int deltaX = planePosCurr.x - planePosPrev.x;
            if (deltaX > 0 && abs(deltaX) < maxPlaneDisplacement) // find the direction of flight
            {
                planeDirection = 1;
                movingTxt = ">>>";
                if (planePosCurr.x > poleCenterMass.x && planePosPrev.x < poleCenterMass.x) {
                    drawCrossingSign(frame, Point(poleCenterMass.x, 0), Point(poleCenterMass.x, S.height));
                }
            } else if (deltaX < 0 && abs(deltaX) < maxPlaneDisplacement) {
                planeDirection = -1;
                movingTxt = "<<<";
                if (planePosCurr.x < poleCenterMass.x && planePosPrev.x > poleCenterMass.x) {
                   drawCrossingSign(frame, Point(poleCenterMass.x, 0), Point(poleCenterMass.x, S.height));
                }
            } else {
                planeDirection = 0;
                movingTxt = "";
            }
            
            ostringstream text;
            text << movingTxt << "  PLANE ON FRAME  " << movingTxt;
            matPrint(frame, Point(1, S.width/2), Scalar(0), text.str());
            drawContours(frame, planeContours, -1, Scalar(0,0,255), 1, LINE_AA);
        }
        
        // imshow("Detected planes", frame);
        // imshow("frameSegmentedPlane", frameSegmentedPlane);
        // drawArrows(frame, pts, nextPts, status, Scalar(255, 0, 0));
        
        for (uint i=0; i<pts.size(); i++) {
            circle(frame, pts[i], 2, Scalar(255,255,0));
        }
        
        frameGray.copyTo(flowPrev); // save frame for optical flow
        imshow("pts", frame);
        
        if (writeOutput)
            outputVideo.write(frame);
        
    }
    
    return 0;
}
