#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <string>
#include <omp.h>
#include <mutex>

using namespace cv;
using namespace std;


const bool GENERATE_VIDEO = false;
const string METHOD = "BM";
const double FPS = 1;

int main(int argc, char* argv[]) {
	//Starting the maximum number of threads
	int nThreads = omp_get_max_threads();
	cout << "Max threads possible " << nThreads << endl;
	omp_set_num_threads(nThreads);
	mutex mtx;
	
	Mat frame1, frame2, gray1, gray2, disp, disp8U, img_roi1, img_roi2;
	vector<Mat> frames1, frames2, disparity;
	Size imageSize = Size(1920, 1080);
	unsigned int nFrames=0, ctr;
	
	//Reading undistorted video files and calculating the number of frames for parallel for
	VideoCapture video1("/home/sourav/Videos/3DU_L1.avi");
	VideoCapture video2("/home/sourav/Videos/3DU_R1.avi");
	video1.set(CV_CAP_PROP_FPS, FPS);
	video2.set(CV_CAP_PROP_FPS, FPS);
	while(video1.isOpened() && video2.isOpened()) {
		video1 >> frame1;
		video2 >> frame2;
		if(frame1.empty() || frame2.empty())
			break;
		else {
			nFrames++;
			frames1.push_back(frame1);
			frames2.push_back(frame2);
			cout << "Reading frame " << nFrames << endl;
		}
	}
	video1.release();
    video2.release();
    
    //Running parallel for using OpenMPI
    #pragma omp parallel for private(frame1,frame2,gray1,gray2,disp,disp8U,img_roi1,img_roi2)
	for(ctr=0; ctr<nFrames; ctr++) {
		int tid = omp_get_thread_num();
		cout << "Thread " << tid << " reading image pair " << ctr << endl;
		frame1 = frames1.at(ctr);
		frame2 = frames2.at(ctr);
		cvtColor(frame1, gray1, CV_BGR2GRAY);
		cvtColor(frame2, gray2, CV_BGR2GRAY);
		
		//Filtering the images
		Rect roi(imageSize.width/5, imageSize.height/5, 4*imageSize.width/5, 4*imageSize.height/5);
		img_roi1 = gray1(roi);
		img_roi2 = gray2(roi);
		bilateralFilter(img_roi1, img_roi1, 5, 50, 50);
		bilateralFilter(img_roi2, img_roi2, 5, 50, 50);
		
		//Choice for methods to calculate disparity
		if(METHOD.compare("BM")) {
			StereoBM bm;
			bm.state->SADWindowSize = 9;
			bm.state->numberOfDisparities = 112;
			bm.state->preFilterSize = 5;
			bm.state->preFilterCap = 61;
			bm.state->minDisparity = -39;
			bm.state->textureThreshold = 507;
			bm.state->uniquenessRatio = 0;
			bm.state->speckleWindowSize = 0;
			bm.state->speckleRange = 8;
			bm.state->disp12MaxDiff = 1;
			bm(img_roi1, img_roi2, disp);
		}
		else if(METHOD.compare("SGBM")) {
			StereoSGBM sgbm;
			sgbm.SADWindowSize = 3;
			sgbm.numberOfDisparities = 144;
			sgbm.preFilterCap = 63;
			sgbm.minDisparity = -39;
			sgbm.uniquenessRatio = 10;
			sgbm.speckleWindowSize = 100;
			sgbm.speckleRange = 32;
			sgbm.disp12MaxDiff = 1;
			sgbm.fullDP = false;
			sgbm.P1 = 216;
			sgbm.P2 = 864;
			sgbm(img_roi1, img_roi2, disp);
		}
		
		normalize(disp, disp8U, 0, 255, CV_MINMAX, CV_8U);
		mtx.lock();
		disparity.push_back(disp8U);
		cout << "Disparity calculated" << endl;
		mtx.unlock();
	}

	//Generating the disparity video
	if(GENERATE_VIDEO) {
		VideoWriter outputVideo("/home/sourav/Videos/Disparity.avi", CV_FOURCC('D','I','V','X'), FPS, imageSize, true);
		for(ctr=0; ctr<disparity.size(); ctr++) {
			outputVideo << disparity.at(ctr);
		}
	}
}
