#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace std;
using namespace cv;

bool gDebug = true;
bool gPause = false;

void onSkipFrames(VideoCapture &cap, int numFrames) {
    int frames = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
    int curFrame = (int)cap.get(CV_CAP_PROP_POS_FRAMES);
    
    int newFrame = curFrame + numFrames;
    
    if (newFrame > frames || newFrame < 1)
        curFrame = numFrames = 0;
    
    if (curFrame <= 370) {
        newFrame = 390;
    } else if (curFrame > 370 && curFrame < 950) {
        newFrame = 950;
    }

    cap.set( CAP_PROP_POS_FRAMES, newFrame );
    
    if (gDebug)
        cout << cap.get(CAP_PROP_POS_FRAMES) << endl;
}

static void convertToGrey(InputArray _in, OutputArray _out) {
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
 * @brief Threshold input image using adaptive thresholding
 */
static void segment(InputArray _in, OutputArray _out) {
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
 */
double getOrientation(vector<Point> &pts, Mat &img, const Point offset) {
    //Construct a buffer used by the pca analysis
    Mat data_pts = Mat((uint)pts.size(), 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i) {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
    
    //Store the position of the object
    Point pos = Point(pca_analysis.mean.at<double>(0, 0) + offset.x,
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
    circle(img, pos, 3, Scalar(255, 0, 255), 2);
    
    Point start(pos - 0.5 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]));
    Point end(pos + 0.5 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]));
    
    line(img, start, end, CV_RGB(255, 255, 0));
    line(img, pos, pos + 0.02 * Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]) , CV_RGB(0, 255, 255));
    
    return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}

static void detectPlanes(InputArray _in,
                        vector< vector< Point > > &_candidates) {
    
    int minPerimeterPoints = 50;
    int maxPerimeterPoints = 3000;
    
    Mat contoursImg;
    _in.getMat().copyTo(contoursImg);
    
    Mat temp;
    contoursImg.copyTo(temp);
    
    vector< vector< Point > > _contours;
    findContours(contoursImg, _contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    for(unsigned int i = 0; i < _contours.size(); i++) {
        
        // check amount of perimeter points
        if(_contours[i].size() < minPerimeterPoints || _contours[i].size() > maxPerimeterPoints) continue;

        
        Rect bounding = boundingRect(_contours[i]);

        // check if the aspect ratio is valid - something is too long...
        float aspectRatio = (float)bounding.width / bounding.height;
        if (aspectRatio > 5) continue;
        
        // check position in frame - ignore if too low
        if (bounding.y > _in.rows()*.8) continue;
        
        
        double arcLen = arcLength(_contours[i], true);
        double conArea = contourArea(_contours[i]);

        if (conArea < 200 || conArea > 10000) {
            cout << "por area! " << conArea << endl;
            drawContours(temp, _contours, i, Scalar(0));
            imshow("temp", temp);
            continue;
        }
//        if (cArea < 300 || cArea)
//        cout << "i: " << i << " - solidity: " << solidity << endl;
        
//        if (isContourConvex(_contours[i])) continue;
        
        _candidates.push_back(_contours[i]);
    }
}

/**
 * @brief Threshold input image using adaptive thresholding
 */
static void segmentPlane(InputArray _in, OutputArray _out) {
    // THRESHOLD
     threshold(_in, _out, 200, 255, THRESH_BINARY | THRESH_OTSU);
    
//    adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 5);
    
    // ERODE / DILATE
    int morph_size = 1;
    Mat elErode = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ) );
    erode(_out, _out, elErode, Point(-1, -1), 8, BORDER_DEFAULT);
    
    Mat elDilate = getStructuringElement( MORPH_RECT, Size( 2*morph_size+1, 2*morph_size+1 ) );
    dilate(_out, _out, elDilate, Point(-1, -1), 6, BORDER_DEFAULT);
    
    
    // INVERT
    bitwise_not(_out, _out);
}

/**
 * @brief Detect the flag pole
 */
static void detectPole(InputArray _in,
                       vector< vector< Point > > &_candidates) {
    
    int minPerimeterPoints = 450;
    int maxPerimeterPoints = 600;
    
    Mat contoursImg;
    _in.getMat().copyTo(contoursImg);
    
    vector< vector< Point > > _contours;
    findContours(contoursImg, _contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    Mat displayOrientation;
    displayOrientation.create( _in.size(), CV_8UC3 );
    
    for(unsigned int i = 0; i < _contours.size(); i++) {
        
        // check if image goes from top to bottom
        Rect bounding = boundingRect(_contours[i]);
        if (bounding.y > 1 || bounding.height < (_in.getMat().rows-2) ) { // the -2 is to fit the contour inside the frame
            if (gDebug)
                cout << "i: " << i << " - excluded by bounding limits" << endl;
            continue;
        }
        
        // check if the aspect ratio is valid
        float aspectRatio = (float)bounding.width / bounding.height;
        if (aspectRatio > 0.2) {
            if (gDebug)
                cout << "i: " << i << " - excluded by aspect ratio: " << aspectRatio << endl;
            continue;
        }
        
        // check perimeter
        if(_contours[i].size() < minPerimeterPoints || _contours[i].size() > maxPerimeterPoints) continue;
        
        // check is square and is convex
        double arcLen = arcLength(_contours[i],true);
        vector< Point > approxCurve;
        approxPolyDP(_contours[i], approxCurve, arcLen * 0.005, true);
        if(approxCurve.size() < 4 || approxCurve.size() > 6 || !isContourConvex(approxCurve)) {
            if (gDebug)
                cout << "i: " << i << " - excluded by approx. size: " << approxCurve.size() << endl;
            continue;
        }
        _candidates.push_back(_contours[i]);
    }
}

static void help(char exe[])
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
    bool saveOutput = false;
    string input;                       // input video

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--video") {
            input = argv[++i];
        } else if (string(argv[i]) == "--output") {
            output = argv[++i];
        } else if (string(argv[i]) == "--help") {
            help(argv[0]);
            return -1;
        } else {
            cout << "Parametro desconhecido: " << argv[i] << endl;
            return -1;
        }
    }

    Mat frame, frameGray, frameSegmentedPole, frameSegmentedPlane;

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

    for (;;) {
        
        char key = (char)waitKey(10); // 10ms/frame
        if(key == 27) break;
        
        switch (key)
        {
            case '[':
                onSkipFrames(inputVideo, -25);
                break;
            case ']':
                onSkipFrames(inputVideo, 50);
                break;
            case 'd':
                gDebug = !gDebug;
                cout << "debug=" << (gDebug?"ON":"OFF") << endl;
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

        
        convertToGrey(frame, frameGray);
        
        
        segment(frameGray, frameSegmentedPole);

        // ISOLATE THE FLAG POLE
        Rect roiRect(roiTL, roiBR);
        Mat roi(frameSegmentedPole, roiRect);

        // FIND POLE CONTOURS
        /*
        Mat contoursImg;
        roi.copyTo(contoursImg);
        
        vector< vector< Point > > poleContours;
        detectPole(contoursImg, poleContours);

        for(unsigned int i = 0; i < poleContours.size(); i++) {
            getOrientation(poleContours[i], frame, roiTL);
        }

        drawContours(frame, poleContours, -1, Scalar(0,0,255), 1, LINE_8, noArray(), INT_MAX, roiTL);
        imshow("Detected candidates", frame);
        */
        
        // DETECT AIRPLANE
        
        segmentPlane(frameGray, frameSegmentedPlane);
    
        vector< vector< Point > > planeContours;
        detectPlanes(frameSegmentedPlane, planeContours);
        
        /*
        cv::Mat mask(frameSegmentedPlane.rows, frameSegmentedPlane.cols, CV_8UC1, Scalar(0));
        drawContours(mask, planeContours, -1, Scalar(255), CV_FILLED);
        imshow("mask", mask);
        */
        
        
        drawContours(frame, planeContours, -1, Scalar(0,0,255), 1, LINE_8);
        imshow("Detected planes", frame);
        imshow("frameSegmentedPlane", frameSegmentedPlane);
        
    }

    return 0;
}
