#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "deck.hpp"

using namespace std;
using namespace cv;

bool gDebug = true;
vector<Point2f> gDeckPosition;

int gThresh = 50;
int gLines = 1;

static void convertToGrey(InputArray _in, OutputArray _out) {
    _out.create(_in.getMat().size(), CV_8UC1);
    if(_in.getMat().type() == CV_8UC3)
        cvtColor(_in.getMat(), _out.getMat(), COLOR_BGR2GRAY);
    else
        _in.getMat().copyTo(_out);
}


/**
 * @brief Finds the intersection of two lines. The lines are defined by (l1p1, l1p2) and (l2p1, p2).
 */
bool intersection(Point2f l1p1, Point2f l1p2, Point2f l2p1, Point2f l2p2, Point2f &r) {
    Point2f x = l2p1 - l1p1;
    Point2f d1 = l1p2 - l1p1;
    Point2f d2 = l2p2 - l2p1;
    
    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;
    
    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = l1p1 + d1 * t1;
    return true;
}

void _storeDeckPosition(Point2f pos) {
    gDeckPosition.push_back(pos);
}

/**
 * @brief Draw detected features
 */
void drawDetectedDecks(InputOutputArray image,
                       InputArrayOfArrays corners,
                       Scalar borderColor = Scalar(0, 255, 0),
                       int width = 3) {
    
    int nMarkers = (int)corners.total();
    for(int i = 0; i < nMarkers; i++) {
        
        Mat currentMarker = corners.getMat(i);
        
        for(int j = 0; j < 4; j++) {
            Point2f p0, p1;
            p0 = currentMarker.ptr< Point2f >(0)[j];
            p1 = currentMarker.ptr< Point2f >(0)[(j + 1) % 4];
            line(image, p0, p1, borderColor, width);
        }
        
        Point2f p0 = currentMarker.ptr< Point2f >(0)[0];
        Point2f p1 = currentMarker.ptr< Point2f >(0)[1];
        Point2f p2 = currentMarker.ptr< Point2f >(0)[2];
        Point2f p3 = currentMarker.ptr< Point2f >(0)[3];
        Point2f center;
        
        line(image, p0, p2, borderColor, width);
        line(image, p1, p3, borderColor, width);
        
        
        double area = contourArea(currentMarker);
        double sideLen = sqrt(area);
        double radius = sideLen * .05; // center circle is aprox. 5% the len of a side
        
        if ( intersection(p0, p2, p1, p3, center) ) {
            circle(image, center, radius, Scalar(0,0,255), 2);
            _storeDeckPosition(center);
        }
    }
}

/**
 * @brief Validate the inside of the detected decks
 */
bool _validateDeck(const Mat& src_gray) {
    
    int cannyThreshold = 70;
    int accumulatorThreshold = 35;
    int minRadius = (src_gray.rows * 0.55)/2; // 55%
    int maxRadius = (src_gray.rows * 0.75)/2; // 75%
    
    std::vector<Vec3f> circles; // will hold the results of the detection
    
    HoughCircles( src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows/2, cannyThreshold, accumulatorThreshold, minRadius, maxRadius );
    
    if (gDebug) { // draw the perspective corrected image with the detected circle
        for( size_t i = 0; i < circles.size(); i++ ) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            circle( src_gray, center, 3, Scalar(0,255,0), -1, 8, 0 );        // circle center
            circle( src_gray, center, radius, Scalar(0,0,255), 3, 8, 0 );    // circle outline
            imshow("hough", src_gray);
        }
    }

    if (circles.size() == 1) {
        Mat canny, cdst;
        Canny(src_gray, canny, 50, 200, 3);
        int marginClip = 8;
        Mat srcHoughLines(canny, Rect(marginClip, marginClip, canny.rows - marginClip*2, canny.cols - marginClip*2) );
        cvtColor(srcHoughLines, cdst, CV_GRAY2BGR);
        
        vector<Vec2f> lines;
        HoughLines(srcHoughLines, lines, 1, gLines * CV_PI/180, gThresh, 0, 0 );
        
        int xLines = 0;
        for( size_t i = 0; i < lines.size(); i++ ) {
            float rho = lines[i][0], theta = lines[i][1];
            if( (theta > CV_PI/180 * 125 && theta < CV_PI / 180 * 145) || (theta > CV_PI/180 * 35 && theta < CV_PI / 180 * 55) ) {
                xLines++;
            }
            
            if (gDebug) {
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;
                pt1.x = cvRound(x0 + 1000*(-b));
                pt1.y = cvRound(y0 + 1000*(a));
                pt2.x = cvRound(x0 - 1000*(-b));
                pt2.y = cvRound(y0 - 1000*(a));
                line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
            }
        }
        
        float linesAtExpectedAngle = (float)xLines/lines.size();        
        if ( linesAtExpectedAngle >= 0.95 ) {
            if (gDebug) {
                imshow("cdst", cdst);
            }
            return true;
        }
    }
    
    return false;
}

/**
 * @brief Given an input image and candidate corners, extract the bits of the candidate
 */
static Mat _removePerspective(InputArray image,
                              InputArray corners) {

    Mat resultImg;
    int resultImgSize = 128;
    Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr< Point2f >(0)[0] = Point2f(0, 0);
    resultImgCorners.ptr< Point2f >(0)[1] = Point2f((float)resultImgSize - 1, 0);
    resultImgCorners.ptr< Point2f >(0)[2] =
    Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
    resultImgCorners.ptr< Point2f >(0)[3] = Point2f(0, (float)resultImgSize - 1);
    
    // remove perspective
    Mat transformation = getPerspectiveTransform(corners, resultImgCorners);
    warpPerspective(image, resultImg, transformation, Size(resultImgSize, resultImgSize), INTER_NEAREST);
    
    return resultImg;
}

/**
 * @brief Threshold input image using adaptive thresholding
 */
static void _threshold(InputArray _in, OutputArray _out, int winSize) {
    if(winSize % 2 == 0) winSize++; // win size must be odd
    int adaptiveThreshConstant = 7;
    adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, winSize, adaptiveThreshConstant);
}

/**
 * @brief Tries to identify one candidate given the dictionary
 */
static bool _identifyOneCandidate(InputArray image,
                                  InputOutputArray corners) {
    
    Mat flatCandidate =  _removePerspective(image, corners);
    
    threshold(flatCandidate, flatCandidate, 125, 255, THRESH_BINARY | THRESH_OTSU);    
    GaussianBlur(flatCandidate, flatCandidate, Size(9, 9), 2, 2 );
    bool detected = _validateDeck(flatCandidate);
    return detected;
}

/**
 * ParallelLoopBody class for the parallelization of the marker identification step
 * Called from function _identifyCandidates()
 */
class IdentifyCandidatesParallel : public ParallelLoopBody {
public:
    IdentifyCandidatesParallel(const Mat *_grey, InputArrayOfArrays _candidates,
                               InputArrayOfArrays _contours,
                               vector< char > *_validCandidates)
    : grey(_grey), candidates(_candidates), contours(_contours), validCandidates(_validCandidates) {}
    
    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;
        
        for(int i = begin; i < end; i++) {
            Mat currentCandidate = candidates.getMat(i);
            if(_identifyOneCandidate(*grey, currentCandidate)) {
                (*validCandidates)[i] = 1;
            }
        }
    }
    
private:
    IdentifyCandidatesParallel &operator=(const IdentifyCandidatesParallel &); // to quiet MSVC
    
    const Mat *grey;
    InputArrayOfArrays candidates, contours;
    vector< char > *validCandidates;
};


/**
 * @brief Given a tresholded image, find the contours, calculate their polygonal approximation
 * and take those that accomplish some conditions
 */
static void _findMarkerContours(InputArray _in, vector< vector< Point2f > > &candidates,
                                vector< vector< Point > > &contoursOut) {
    
    double minPerimeterRate = 0.03;
    double maxPerimeterRate = 4.0;
    double minCornerDistanceRate = 0.05;
    
    // calculate maximum and minimum sizes in pixels
    unsigned int minPerimeterPixels =
    (unsigned int)(minPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));
    unsigned int maxPerimeterPixels =
    (unsigned int)(maxPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));
    
    Mat contoursImg;
    _in.getMat().copyTo(contoursImg);
    vector< vector< Point > > contours;
    
    findContours(contoursImg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    // now filter list of contours
    for(unsigned int i = 0; i < contours.size(); i++) {
        // check perimeter
        if(contours[i].size() < minPerimeterPixels || contours[i].size() > maxPerimeterPixels)
            continue;
        
        // check is square and is convex
        vector< Point > approxCurve;
        approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.03, true);
        if(approxCurve.size() != 4 || !isContourConvex(approxCurve)) continue;
        
        // check min distance between corners
        double minDistSq =
        max(contoursImg.cols, contoursImg.rows) * max(contoursImg.cols, contoursImg.rows);
        for(int j = 0; j < 4; j++) {
            double d = (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
            (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
            (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
            (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y);
            minDistSq = min(minDistSq, d);
        }
        double minCornerDistancePixels = double(contours[i].size()) * minCornerDistanceRate;
        if(minDistSq < minCornerDistancePixels * minCornerDistancePixels) continue;
        
        // if it passes all the test, add to candidates vector
        vector< Point2f > currentCandidate;
        currentCandidate.resize(4);
        for(int j = 0; j < 4; j++) {
            currentCandidate[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
        }
        candidates.push_back(currentCandidate);
        contoursOut.push_back(contours[i]);
    }
}


/**
 * ParallelLoopBody class for the parallelization of the basic candidate detections using
 * different threhold window sizes. Called from function _detectInitialCandidates()
 */
class DetectInitialCandidatesParallel : public ParallelLoopBody {
public:
    DetectInitialCandidatesParallel(const Mat *grey,
                                    vector< vector< vector< Point2f > > > *candidatesArrays,
                                    vector< vector< vector< Point > > > *contoursArrays)
    : grey(grey), candidatesArrays(candidatesArrays), contoursArrays(contoursArrays) {}
    
    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;
        
        int adaptiveThreshWinSizeMin = 3;
        int adaptiveThreshWinSizeStep = 10;
        
        for(int i = begin; i < end; i++) {
            int currScale = adaptiveThreshWinSizeMin + i * adaptiveThreshWinSizeStep;
            // threshold
            Mat thresh;
            _threshold(*grey, thresh, currScale);
            
            // detect rectangles
            _findMarkerContours(thresh, (*candidatesArrays)[i], (*contoursArrays)[i]);
        }
    }
    
private:
    DetectInitialCandidatesParallel &operator=(const DetectInitialCandidatesParallel &);
    
    const Mat *grey;
    vector< vector< vector< Point2f > > > *candidatesArrays;
    vector< vector< vector< Point > > > *contoursArrays;
};


/**
 * @brief Initial steps on finding square candidates
 */
static void _detectInitialCandidates(const Mat &grey,
                                     vector< vector< Point2f > > &candidates,
                                     vector< vector< Point > > &contours) {
    
    int adaptiveThreshWinSizeMin = 3;
    int adaptiveThreshWinSizeMax = 23;
    int adaptiveThreshWinSizeStep = 10;
    
    // number of window sizes (scales) to apply adaptive thresholding
    int nScales = (adaptiveThreshWinSizeMax - adaptiveThreshWinSizeMin) / adaptiveThreshWinSizeStep + 1;
    
    vector< vector< vector< Point2f > > > candidatesArrays((size_t) nScales);
    vector< vector< vector< Point > > > contoursArrays((size_t) nScales);
    
    if (gDebug) { // parallelize only if not in debug mode
        //for each value in the interval of thresholding window sizes
        for(int i = 0; i < nScales; i++) {
            int currScale = adaptiveThreshWinSizeMin + i * adaptiveThreshWinSizeStep;
            // treshold
            Mat thresh;
            _threshold(grey, thresh, currScale);
            
//            imshow("thresh", thresh);
            
            // detect rectangles
            _findMarkerContours(thresh, candidatesArrays[i], contoursArrays[i]);
        }
    
    } else {
        parallel_for_(Range(0, nScales), DetectInitialCandidatesParallel(&grey, &candidatesArrays, &contoursArrays));
    }
    
    // join candidates
    for(int i = 0; i < nScales; i++) {
        for(unsigned int j = 0; j < candidatesArrays[i].size(); j++) {
            candidates.push_back(candidatesArrays[i][j]);
            contours.push_back(contoursArrays[i][j]);
        }
    }
}


/**
 * @brief Detect square candidates in the input image
 */
static void _detectCandidates(InputArray image,
                              OutputArrayOfArrays _candidates,
                              OutputArrayOfArrays _contours) {
    
    Mat grey;
    image.copyTo(grey);
    
    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contoursOut;
    
    /// 1. DETECT FIRST SET OF CANDIDATES
    _detectInitialCandidates(grey, candidates, contoursOut);
    
    /// 2. PARSE OUTPUT
    _candidates.create((int)candidates.size(), 1, CV_32FC2);
    _contours.create((int)contoursOut.size(), 1, CV_32SC2);
    for(int i = 0; i < (int)candidates.size(); i++) {
        _candidates.create(4, 1, CV_32FC2, i, true);
        Mat m = _candidates.getMat(i);
        for(int j = 0; j < 4; j++)
            m.ptr< Vec2f >(0)[j] = candidates[i][j];
        
        _contours.create((int)contoursOut[i].size(), 1, CV_32SC2, i, true);
        Mat c = _contours.getMat(i);
        for(unsigned int j = 0; j < contoursOut[i].size(); j++)
            c.ptr< Point2i >()[j] = contoursOut[i][j];
    }
}


/**
 * @brief Identify possible decks
 */
static void _identifyCandidates(InputArray image,
                                InputArrayOfArrays _candidates,
                                InputArrayOfArrays _contours,
                                OutputArrayOfArrays _accepted,
                                OutputArrayOfArrays _rejected = noArray()) {
    
    int ncandidates = (int)_candidates.total();
    
    vector< Mat > accepted;
    vector< Mat > rejected;
    
    Mat grey = image.getMat();
    
    vector< char > validCandidates(ncandidates, 0);
    
    // Analyze each of the candidates
    if (gDebug) { // parallelize only if not in debug mode
        for (int i = 0; i < ncandidates; i++) {
            Mat currentCandidate = _candidates.getMat(i);
            if (_identifyOneCandidate(grey, currentCandidate)) {
                validCandidates[i] = 1;
            }
        }
    } else {
        parallel_for_(Range(0, ncandidates), IdentifyCandidatesParallel(&grey, _candidates, _contours, &validCandidates));
    }
    
    for(int i = 0; i < ncandidates; i++) {
        if(validCandidates[i] == 1) {
            accepted.push_back(_candidates.getMat(i));
        } else {
            rejected.push_back(_candidates.getMat(i));
        }
    }
    
    // parse output
    _accepted.create((int)accepted.size(), 1, CV_32FC2);
    
    for (unsigned int i = 0; i < accepted.size(); i++) {
        _accepted.create(4, 1, CV_32FC2, i, true);
        Mat m = _accepted.getMat(i);
        accepted[i].copyTo(m);
    }
}

void detectDecks(InputArray image,
                 OutputArrayOfArrays corners,
                 OutputArrayOfArrays _rejectedImgPoints
                 ) {
    
    /// STEP 1: Detect deck candidates
    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contours;
    
    _detectCandidates(image, candidates, contours);
    
    /// STEP 2: Check candidate
    _identifyCandidates(image, candidates, contours, corners, _rejectedImgPoints);
    
    if (gDebug) { // show image with possible candidates
        Mat temp;
        image.copyTo(temp);
        cvtColor(temp, temp, COLOR_GRAY2BGR);
        drawContours(temp, contours, -1, Scalar(0,0,255), 3);
        imshow("detect candidates", temp);
    }
}

static void help()
{
    cout
    << "------------------------------------------------------------------------------" << endl
    << "DAI . DETECAO E ANALISE DE IMAGENS"                                             << endl
    << "2016/2o. SEMESTRE"                                                              << endl
    << "ALUNO: CASSIANO RABELO E SILVA"                                                 << endl
    << "QUESTAO #3 . AEROMODELOS"                                               << endl << endl
    << "Utilizacao:"                                                                    << endl
    << "dai-questao-03.exe --video <video> [--output <video>]"                          << endl
    << "------------------------------------------------------------------------------" << endl
    << "Utilizando OpenCV " << CV_VERSION << endl << endl;
}

/////////////////////////////////////
int main(int argc, char *argv[]) {
    
    if (argc == 1) {
        help();
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
            help();
            return -1;
        } else {
            cout << "Parametro desconhecido: " << argv[i] << endl;
            return -1;
        }
    }
    
    Mat frame, frameGray;
    
    VideoCapture inputVideo;
    inputVideo.open(input);
    CV_Assert(inputVideo.isOpened());
    
    // output resolution based on input
    Size S = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),
                  (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));

    cout << S.width << " - " << S.height << endl;
    
    for (;;) {
        inputVideo >> frame;
        if (frame.empty()) {
            break;
        }
        
        convertToGrey(frame, frameGray);
        
//        Canny(frameGray, frameGray, 50, 200, 3);
//        adaptiveThreshold(frameGray, frameGray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 7, 10);

        
        if (/* DISABLES CODE */ (false)) {
//            Sobel(frameGray, frameGray, CV_8U, 1, 1, 3, 5, 0, BORDER_DEFAULT);
//            bitwise_not(frameGray, frameGray);
//            GaussianBlur(frameGray, frameGray, Size(5,5), 0);
            adaptiveThreshold(frameGray, frameGray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 23, 7);
//            threshold(frameGray, frameGray, 230, 255, THRESH_BINARY);
            
//            float linesAtExpectedAngle = (float)xLines/lines.size();
//            if ( linesAtExpectedAngle >= 0.95 ) {
//                if (gDebug) {
//                    imshow("cdst", cdst);
//                }
//                return true;
//            }

            imshow("adaptiveThreshold", frameGray);
            
        } else if (/* DISABLES CODE */ (false)) {
            Canny(frameGray, frameGray, 200, 255, 3);
            imshow("Canny", frameGray);
            
        } else if(/* DISABLES CODE */ (true)) {
            
            
            // MASCARA
//            Mat mask(frameGray.size(), frameGray.type(), Scalar(0) );
//            rectangle(mask, Point(0,0), Point(frameGray.cols*0.3, frameGray.rows), Scalar(0));
//            imshow("mask", mask);
            
            
//            Mat roi(mask, Rect(mask.cols * 0.3, mask.rows*0.3, mask.cols.mask.rows/2 ) );
//            roi = Scalar(20, 20, 20); // a mascara não aceita tons de cinza...
//            src.copyTo( dst, mask);
            
            
            // THRESHOLD
            threshold(frameGray, frameGray, 125, 255, THRESH_BINARY | THRESH_OTSU);

            // ERODE / DILATE
            int morph_size = 1;
            Mat elErode = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size+1, 2*morph_size+1 ) );
            erode(frameGray, frameGray, elErode, Point(-1, -1), 4, BORDER_DEFAULT);

            Mat elDilate = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size+1, 2*morph_size+1 ) );
            dilate(frameGray, frameGray, elDilate, Point(-1, -1), 4, BORDER_DEFAULT);
            
            // INVERT
            bitwise_not(frameGray, frameGray);
            
            Rect maskCoords( Point(frameGray.cols*0.3, frameGray.rows*0.35), Point(frameGray.cols*0.7, frameGray.rows*0.7) );
            Mat masked(frameGray, maskCoords );
            
            Mat frameGrayMasked;
            masked.copyTo(frameGrayMasked);
            imshow("frameGrayMasked", frameGrayMasked);
            
            
            // HOUGHLINES
            Mat houghlines;
            frameGray.copyTo(houghlines);
            vector<Vec2f> lines;
            
            // sem usar canny
            // gThresh = 310
            // gLines = 5
            Canny(houghlines, houghlines, 50, 200, 3);
//            imshow("Canny", houghlines);
            
//            int gThresh = 30;
//            int gLines = 10;
            HoughLines(houghlines, lines, 1, gLines * CV_PI/180, gThresh, 0, 0);
            
            int xLines = 0;
            for( size_t i = 0; i < lines.size(); i++ ) {
                float rho = lines[i][0], theta = lines[i][1];
                
                if( (theta > CV_PI/180 * 160) || (theta < CV_PI / 180 * 20) ) {
                    xLines++;
                    Point pt1, pt2;
                    double a = cos(theta), b = sin(theta);
                    double x0 = a*rho, y0 = b*rho;
                    pt1.x = cvRound(x0 + 1000*(-b));
                    pt1.y = cvRound(y0 + 1000*(a));
                    pt2.x = cvRound(x0 - 1000*(-b));
                    pt2.y = cvRound(y0 - 1000*(a));
                    line( frame, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
                }
                
            }
            
            
            
//            float linesAtExpectedAngle = (float)xLines/lines.size();
//            if ( linesAtExpectedAngle >= 0.95 ) {
//                if (gDebug) {
//                    imshow("cdst", cdst);
//                }
//                return true;
//            }
            
//            Mat contoursImg;
//            vector< vector< Point > > contours;
//            findContours(frameGray, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
            
//            cout << contours.size() << endl;
            
//            Mat temp;
//            frame.copyTo(temp);
//            drawContours(temp, contours, -1, Scalar(0,0,255), 1);
//            imshow("detect candidates", temp);
            
//            imshow("dilate / erode", frameGray);
            imshow("Frame", frame);
        }
        
        
        
        
        

        
//        imshow("Canny", frameGray);
        
        char key = (char)waitKey(10); // 10ms/frame
        if(key == 27) break;
        
        switch (key)
        {
            case 'd':
                gDebug = !gDebug;
                cout << "debug=" << (gDebug?"ON":"OFF") << endl;
                break;
                
            case ']':
                gThresh++;
                cout << "gThresh: " << gThresh << endl;
                break;
                
            case '[':
                gThresh--;
                cout << "gThresh: " << gThresh << endl;
                break;
                
            case 'p':
                gLines++;
                cout << "gLines: " << gLines << endl;
                break;
                
            case 'o':
                gLines--;
                cout << "gLines: " << gLines << endl;
                break;
        }
    }
    
    return 0;
}
