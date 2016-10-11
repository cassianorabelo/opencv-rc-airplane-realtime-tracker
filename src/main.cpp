#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

bool gDebug = true;
vector<Point2f> gDeckPosition;

int gThresh = 50;
int gLines = 1;

RNG rng(12345);

static void convertToGrey(InputArray _in, OutputArray _out) {
    _out.create(_in.getMat().size(), CV_8UC1);
    if(_in.getMat().type() == CV_8UC3)
        cvtColor(_in.getMat(), _out.getMat(), COLOR_BGR2GRAY);
    else
        _in.getMat().copyTo(_out);
}


/**
 * @brief Detect square candidates in the input image
 */
static void _detectPole(InputArray image,
                        OutputArrayOfArrays _candidates,
                        OutputArrayOfArrays _contours) {
    
    Mat grey;
    image.copyTo(grey);
    
    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contoursOut;
    
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

    // ROI top left and bottom right in respect to the full frame
    Point roiTL = Point(S.width*0.3, S.height*0.35);
    Point roiBR = Point(S.width*0.7, S.height*0.7);

    for (;;) {
        inputVideo >> frame;
        if (frame.empty()) {
            break;
        }

        convertToGrey(frame, frameGray);

        // THRESHOLD
        threshold(frameGray, frameGray, 125, 255, THRESH_BINARY | THRESH_OTSU);

        // ERODE / DILATE
        int morph_size = 1;
        Mat elErode = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size+1, 2*morph_size+1 ) );
        erode(frameGray, frameGray, elErode, Point(-1, -1), 6, BORDER_DEFAULT);

        Mat elDilate = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size+1, 2*morph_size+1 ) );
        dilate(frameGray, frameGray, elDilate, Point(-1, -1), 4, BORDER_DEFAULT);

        // INVERT
        bitwise_not(frameGray, frameGray);

        // ISOLATE POLE
        Rect roiRect(roiTL, roiBR);
        Mat roi(frameGray, roiRect);
        
        if (gDebug)
            imshow("ROI", roi);

        // FIND POLE CONTOURS
        Mat contoursImg;
        roi.copyTo(contoursImg);
        vector< vector< Point > > contours;
        vector< vector< Point > > candidates;
        findContours(contoursImg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        int minPerimeterPoints = 450;
        int maxPerimeterPoints = 600;

        Mat displayOrientation;
        displayOrientation.create( roi.size(), frame.type() );
        
        for(unsigned int i = 0; i < contours.size(); i++) {

            // check if image goes from top to bottom
            Rect bounding = boundingRect(contours[i]);
            if (bounding.y > 1 || bounding.height < (roi.rows-2) ) { // the -2 is to fit the contour inside the frame
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
            if(contours[i].size() < minPerimeterPoints || contours[i].size() > maxPerimeterPoints) continue;

            // check is square and is convex
            double arcLen = arcLength(contours[i],true);
            vector< Point > approxCurve;
            approxPolyDP(contours[i], approxCurve, arcLen * 0.005, true);
            if(approxCurve.size() < 4 || approxCurve.size() > 6 || !isContourConvex(approxCurve)) {
                if (gDebug)
                    cout << "i: " << i << " - excluded by approx. size: " << approxCurve.size() << endl;
                continue;
            }

            candidates.push_back(contours[i]);
        }

        for(unsigned int i = 0; i < candidates.size(); i++) {
            getOrientation(candidates[i], frame, roiTL);
        }

        drawContours(frame, candidates, -1, Scalar(0,0,255), 1, LINE_8, noArray(), INT_MAX, roiTL);
        imshow("Detected candidates", frame);

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
