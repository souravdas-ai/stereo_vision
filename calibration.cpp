#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>
#include <omp.h>
#include <mutex>

using namespace cv;
using namespace std;

const bool CALIBRATE = false;
const Size imageSize = Size(1920, 1080);
const unsigned int INTERVAL = 20;
const double FPS = 10;


int main(int argc, char* argv[]) {
	//Starting the maximum number of threads
	int nThreads = omp_get_max_threads();
	cout << "Max threads possible " << nThreads << endl;
	omp_set_num_threads(nThreads);
	mutex mtx;
	
	Mat frame1(imageSize, CV_8UC3);
	Mat frame2(imageSize, CV_8UC3);
	Mat CM1 = Mat(3, 3, CV_64FC1);
	Mat CM2 = Mat(3, 3, CV_64FC1);
	Mat D1, D2;
	Mat R, T, E, F;
	Mat R1, R2, P1, P2, Q;
	
	if(CALIBRATE) {
	//Setting the checkerboard parameters
	const double squareSize = 0.108;
	const int boardWidth = 8;
	const int boardHeight = 6;
	Size boardSize = Size(boardWidth, boardHeight);
	
	Mat gray1, gray2;
	vector<Mat> frames1, frames2;
	vector<vector<Point2f> > imagePoints1;
	vector<vector<Point2f> > imagePoints2;
    vector<vector<Point3f> > objectPoints;
    vector<Point2f> corners1;
    vector<Point2f> corners2;
    vector<Point3f> obj;
    
    for(int i=0; i < boardSize.height; i++)
		for(int j=0; j < boardSize.width; j++)
			obj.push_back(Point3f(i*squareSize, j*squareSize, 0));
    
    bool found1=false, found2=false;
    unsigned int nFrames=0, ctr=1;
	
	//Reading stereo videos at a fixed FPS and calculating the number of frames for parallel for
	VideoCapture video1("/home/sourav/Videos/raw_data/3D_L1.MP4");
	VideoCapture video2("/home/sourav/Videos/raw_data/3D_R1.MP4");
	video1.set(CV_CAP_PROP_FPS, FPS);
	video2.set(CV_CAP_PROP_FPS, FPS);
	
	while(video1.isOpened() && video2.isOpened()) {
		video1 >> frame1;
		video2 >> frame2;
		if(frame1.empty() || frame2.empty())
			break;
		if((double)(ctr%INTERVAL) != 0) {
			ctr++;
			continue;
		}
		else {
			nFrames++;
			ctr++;
			frames1.push_back(frame1);
			frames2.push_back(frame2);
			cout << "Reading frame " << nFrames << endl;
		}
	}
	video1.release();
    video2.release();
	
	//Running parallel for using OpenMPI
	#pragma omp parallel for private(frame1,frame2,gray1,gray2,found1,found2,corners1,corners2)
	for(ctr=0; ctr<nFrames; ctr++) {			
		int tid = omp_get_thread_num();
		cout << "Thread " << tid << " reading image pair " << ctr << endl;
		frame1 = frames1.at(ctr);
		frame2 = frames2.at(ctr);
		cvtColor(frame1, gray1, CV_BGR2GRAY);
		cvtColor(frame2, gray2, CV_BGR2GRAY);
		found1 = findChessboardCorners(gray1, boardSize, corners1,
										CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		found2 = findChessboardCorners(gray2, boardSize, corners2,
										CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		
		//Using static search window size Size(11,11)
		if(found1)
			cornerSubPix(gray1, corners1, Size(11,11), Size(-1,-1),
                         TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
		if(found2)
			cornerSubPix(gray2, corners2, Size(11,11), Size(-1,-1),
                         TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
		if(found1 && found2) {
			mtx.lock();
			imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            objectPoints.push_back(obj);
            cout << "Corners stored" << endl;
            mtx.unlock();
        }
	}
    
    //Finding the intrinsic parameters of the stereo camera
	cout << "Calibration started" << endl;
    double rms = stereoCalibrate(objectPoints, imagePoints1, imagePoints2, 
                    CM1, D1, CM2, D2, imageSize, R, T, E, F,
                    TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1, 1e-5),
                    CV_CALIB_FIX_INTRINSIC);
                    //CV_CALIB_FIX_ASPECT_RATIO +
                    //CV_CALIB_ZERO_TANGENT_DIST +
                    //CV_CALIB_SAME_FOCAL_LENGTH +
                    //CV_CALIB_RATIONAL_MODEL +
                    //CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
	
	FileStorage fs("stereocalib.yml", FileStorage::WRITE);
    fs << "CM1" << CM1;
    fs << "CM2" << CM2;
    fs << "D1" << D1;
    fs << "D2" << D2;
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;
    cout << "Calibration complete with RMS error: " << rms << endl;
    
    //Finding the rectification parameters
    cout << "Rectification started" << endl;
    stereoRectify(CM1, D1, CM2, D2, imageSize, R, T, R1, R2, P1, P2, Q);
    fs << "R1" << R1;
    fs << "R2" << R2;
    fs << "P1" << P1;
    fs << "P2" << P2;
    fs << "Q" << Q;
    cout << "Rectification complete" << endl;
    fs.release();
    }
    
    //Reading calibration info
    if(!CALIBRATE) {
		FileStorage fs("stereocalib.yml", FileStorage::READ);
		if(fs.isOpened()) {
			fs["CM1"] >> CM1;
			fs["CM2"] >> CM2;
			fs["D1"] >> D1;
			fs["D2"] >> D2;
			fs["R"] >> R;
			fs["T"] >> T;
			fs["E"] >> E;
			fs["F"] >> F;
			fs["R1"] >> R1;
			fs["R2"] >> R2;
			fs["P1"] >> P1;
			fs["P2"] >> P2;
			fs["Q"] >> Q;
		}
		else
			cout << "Error in reading from calibration file" << endl;
    
		//Generating undistorted videos
		cout << "Applying undistortion" << endl;
		Mat frameU1(imageSize, CV_8UC3);
		Mat frameU2(imageSize, CV_8UC3);
		int ctr=0;
		Mat map1x, map1y, map2x, map2y;
		initUndistortRectifyMap(CM1, D1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
		initUndistortRectifyMap(CM2, D2, R2, P2, imageSize, CV_32FC1, map2x, map2y);
		cout << "Undistortion complete" << endl;
		VideoWriter outputVideo1("/home/sourav/Videos/3DU_L1.avi", CV_FOURCC('D','I','V','X'), FPS, imageSize, true);
		VideoWriter outputVideo2("/home/sourav/Videos/3DU_R1.avi", CV_FOURCC('D','I','V','X'), FPS, imageSize, true);
    	VideoCapture video1("/home/sourav/Videos/raw_data/3D_L1.MP4");
		VideoCapture video2("/home/sourav/Videos/raw_data/3D_R1.MP4");
		video1.set(CV_CAP_PROP_FPS, FPS);
		video2.set(CV_CAP_PROP_FPS, FPS);
		namedWindow("left", CV_WINDOW_NORMAL);
		namedWindow("right", CV_WINDOW_NORMAL);
    	while(video1.isOpened() && video2.isOpened()) {
			video1 >> frame1;
			video2 >> frame2;
			if(frame1.empty() || frame2.empty())
				break;
    		remap(frame1, frameU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        	remap(frame2, frameU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        	//undistort(frame1, frameU1, CM1, D1, P1);
        	//undistort(frame2, frameU2, CM2, D2, P2);
        	outputVideo1 << frameU1;
        	outputVideo2 << frameU1;
        	imshow("left", frameU1);
        	imshow("right", frameU2);
        	waitKey(30);
        	cout << "Writing frame: " << ctr++ << endl;
        }
        video1.release();
        video2.release();
	}
}
