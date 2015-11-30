/*
 * proposal.cpp
 *
 *  Created on: May 2, 2015
 *      Author: sourav
 */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/contrib/contrib.hpp"

/*
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
*/

#include <iostream>
#include <string>

using namespace cv;
using namespace std;


// Change the path of the video files
const string VIDEO_PATH1 = "/home/sourav/Videos/raw_data/3D_L2.MP4";
const string VIDEO_PATH2 = "/home/sourav/Videos/raw_data/3D_R2.MP4";

const int SETUP = 1;
const bool PREPROCESS = true;
const string METHOD = "SBM";
const bool COLOR = false;
const Size imageSize = Size(1920, 1080);
const int dialation_size = 7;
const int erosion_size = 10;
const int area_thresh = 30000;

Mat CM1, D1, R1, P1;
Mat CM2, D2, R2, P2;
Mat R, T, Q;

void extract_frame() {
  VideoCapture video1(VIDEO_PATH1);
  VideoCapture video2(VIDEO_PATH2);
  Mat frame1, frame2, gray1, gray2, gray_filtered1, gray_filtered2;
  video1.set(CV_CAP_PROP_POS_MSEC, 217000);
  video2.set(CV_CAP_PROP_POS_MSEC, 217000);
  int ctr = 0;

  while ((video1.isOpened() && video2.isOpened())
      && (video1.get(CV_CAP_PROP_POS_MSEC) <= 218000
          && video2.get(CV_CAP_PROP_POS_MSEC) <= 218000)) {
    video1 >> frame1;
    video2 >> frame2;
    if (frame1.empty() || frame2.empty())
      break;
    
    ctr++;
    if (ctr % 2 == 0) {
      cvtColor(frame1, gray1, COLOR_BGR2GRAY);
      cvtColor(frame2, gray2, COLOR_BGR2GRAY);
      
      if (ctr == 2) {
		if (PREPROCESS) {
			cout << "Removing salt-pepper noise" << endl;
			medianBlur(gray1, gray_filtered1, 5);
			medianBlur(gray2, gray_filtered2, 5);
		}
		else {
			gray1.copyTo(gray_filtered1);
			gray2.copyTo(gray_filtered2);
		}
		
        string filename1 = "frame_left"+to_string(SETUP)+".jpg";
      	string filename2 = "frame_right"+to_string(SETUP)+".jpg";
        imwrite(filename1, frame1);
        imwrite(filename2, frame2);

      	filename1 = "gray_left"+to_string(SETUP)+".jpg";
      	filename2 = "gray_right"+to_string(SETUP)+".jpg";
        imwrite(filename1, gray_filtered1);
        imwrite(filename2, gray_filtered2);
      }
    }
    else
		continue;
  }

  destroyAllWindows();
  video1.release();
  video2.release();
}

void undistort_images(Mat image1, Mat image2) {
	Mat map1x, map1y, map2x, map2y;
	Mat undistorted_image1(imageSize, CV_8UC3);
	Mat undistorted_image2(imageSize, CV_8UC3);
	string filename1, filename2;

	filename1 = "camera_left"+to_string(SETUP)+".yml";
	filename2 = "camera_right"+to_string(SETUP)+".yml";
	FileStorage fs1(filename1, FileStorage::READ);
	FileStorage fs2(filename2, FileStorage::READ);

	if(fs1.isOpened() && fs2.isOpened()) {
		fs1["camera_matrix"] >> CM1;
		fs2["camera_matrix"] >> CM2;
		fs1["distortion_coefficients"] >> D1;
		fs2["distortion_coefficients"] >> D2;
		fs1["rectification_matrix"] >> R1;
		fs2["rectification_matrix"] >> R2;
		fs1["projection_matrix"] >> P1;
		fs2["projection_matrix"] >> P2;

		cout << "Applying undistortion" << endl;
		initUndistortRectifyMap(CM1, D1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
		initUndistortRectifyMap(CM2, D2, R2, P2, imageSize, CV_32FC1, map2x, map2y);
		remap(image1, undistorted_image1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(image2, undistorted_image2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        filename1 = "undistorted_left"+to_string(SETUP)+".jpg";
        filename2 = "undistorted_right"+to_string(SETUP)+".jpg";
        imwrite(filename1, undistorted_image1);
        imwrite(filename2, undistorted_image2);
        cout << "Undistortion complete" << endl;
	}
	else
		cout << "Error in reading the calibration files" << endl;
}


void compute_disparity(Mat image1, Mat image2) {
  Mat disp, disp8;
  cout << "Computing disparity" << endl;

  if (METHOD.compare("SBM") == 0) {
    StereoBM sbm;
    sbm.state->SADWindowSize = 9;
    sbm.state->numberOfDisparities = 112;
    sbm.state->preFilterSize = 5;
    sbm.state->preFilterCap = 61;
    sbm.state->minDisparity = -39;
    sbm.state->textureThreshold = 507;
    sbm.state->uniquenessRatio = 0;
    sbm.state->speckleWindowSize = 0;
    sbm.state->speckleRange = 8;
    sbm.state->disp12MaxDiff = 1;
    sbm(image1, image2, disp);
  }
  else if (METHOD.compare("SGBM") == 0) {
    StereoSGBM sgbm;
    sgbm.SADWindowSize = 5;
    sgbm.numberOfDisparities = 192;
    sgbm.preFilterCap = 4;
    sgbm.minDisparity = -64;
    sgbm.uniquenessRatio = 1;
    sgbm.speckleWindowSize = 150;
    sgbm.speckleRange = 2;
    sgbm.disp12MaxDiff = 10;
    sgbm.fullDP = false;
    sgbm.P1 = 600;
    sgbm.P2 = 2400;
    sgbm(image1, image2, disp);
  }

  normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
  cout << "Disparity computed" << endl;

  string filename = "disparity"+to_string(SETUP)+".jpg";
  imwrite(filename, disp8);
  namedWindow("disparity", WINDOW_NORMAL);
  imshow("disparity", disp8);
}

Mat reproject3d() {
    cout << "Reprojecting image to 3D" << endl;
    Mat disparity;
	Mat img3d(imageSize, CV_32FC3);
	string filename;
	
	filename = "relative_pose"+to_string(SETUP)+".yml";
    FileStorage fs(filename, FileStorage::READ);

	if(fs.isOpened()) {
		fs["translation_matrix"] >> T;
		fs["rotation_matrix"] >> R;
	}
	else
		cout << "Error in reading from relative pose file" << endl;
	
    stereoRectify(CM1, D1, CM2, D2, imageSize, R, T, R1, R2, P1, P2, Q, false, CV_32F);
    
    filename = "disparity"+to_string(SETUP)+".jpg";
    disparity = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    reprojectImageTo3D(disparity, img3d, Q);
    cout << "Reprojection complete" << endl;
    return img3d;
}

/*
boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

void create_pointcloud(Mat img3d) {
	//Creating 3D point cloud
	cout << "Creating 3D point clouds" << endl;
	string filename = "frame_left"+to_string(SETUP)+".jpg";
    Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
	double px, py, pz;
	uchar pr, pg, pb;
	for(int i=0; i<img.rows; i++) {
		uchar* rgb_ptr = img.ptr<uchar>(i);
		double* recons_ptr = img3d.ptr<double>(i);
		for(int j=0; j<img.cols; j++) {
			//Get XYZRGB info
			px = recons_ptr[3*j];
			py = recons_ptr[3*j+1];
			pz = recons_ptr[3*j+2];
			pb = rgb_ptr[3*j];
			pg = rgb_ptr[3*j+1];
			pr = rgb_ptr[3*j+2];
			
			//Insert info into point cloud structure
			pcl::PointXYZRGB point;
			point.x = px;
			point.y = py;
			point.z = pz;
			uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
			point.rgb = *reinterpret_cast<float*>(&rgb);
			point_cloud_ptr->points.push_back (point);
		}
	}
  point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
  point_cloud_ptr->height = 1;
  
  //Create visualizer
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer = createVisualizer(point_cloud_ptr);
  
  //Running the visualizer
  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}
*/

int main(int argc, char* argv[]) {
    Mat image1, image2, gray1, gray2;
    Mat bw1, bw2, edge1, edge2;
    Mat element_e, element_d;
    Mat undistorted_image1(imageSize, CV_8UC3);
    Mat undistorted_image2(imageSize, CV_8UC3);
    string filename1, filename2;

    extract_frame();

    if(COLOR) {
    filename1 = "frame_left"+to_string(SETUP)+".jpg";
    filename2 = "frame_right"+to_string(SETUP)+".jpg";
    image1 = imread(filename1, CV_LOAD_IMAGE_COLOR);
    image2 = imread(filename2, CV_LOAD_IMAGE_COLOR);

    threshold(image1, image1, 40, 255, CV_THRESH_TOZERO);
    threshold(image2, image2, 40, 255, CV_THRESH_TOZERO);

    namedWindow("frame_left", WINDOW_NORMAL);
    namedWindow("frame_right", WINDOW_NORMAL);
    imshow("frame_left", image1);
    imshow("frame_right", image2);

    undistort_images(image1, image2);
    filename1 = "undistorted_left"+to_string(SETUP)+".jpg";
    filename2 = "undistorted_right"+to_string(SETUP)+".jpg";
    undistorted_image1 = imread(filename1, CV_LOAD_IMAGE_COLOR);
    undistorted_image2 = imread(filename2, CV_LOAD_IMAGE_COLOR);
    }

    else if(!COLOR) {
    filename1 = "gray_left"+to_string(SETUP)+".jpg";
    filename2 = "gray_right"+to_string(SETUP)+".jpg";
    gray1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    gray2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);

    threshold(gray1, bw1, 40, 255, CV_THRESH_BINARY);
    threshold(gray2, bw2, 40, 255, CV_THRESH_BINARY);

    element_d = getStructuringElement(MORPH_ELLIPSE,
                                        Size(2 * dialation_size + 1, 2 * dialation_size + 1),
                                        Point(dialation_size, dialation_size));
    dilate(bw1, bw1, element_d);
    dilate(bw2, bw2, element_d);

    element_e = getStructuringElement(MORPH_ELLIPSE,
                                        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        Point(erosion_size, erosion_size));
    erode(bw1, bw1, element_e);
    erode(bw2, bw2, element_e);

    vector<vector<Point> > contours1;
    findContours(bw1.clone(), contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (unsigned int i = 0; i < contours1.size(); i++) {
        double area = contourArea(contours1[i]);
        if (area < area_thresh)
            drawContours(bw1, contours1, i, Scalar(0), -1);
    }
    vector<vector<Point> > contours2;
    findContours(bw2.clone(), contours2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (unsigned int i = 0; i < contours2.size(); i++) {
        double area = contourArea(contours2[i]);
        if (area < area_thresh)
            drawContours(bw2, contours2, i, Scalar(0), -1);
    }

    dilate(bw1, bw1, element_e);
    dilate(bw2, bw2, element_e);

    Canny(bw1, edge1, 50, 150, 3);
    Canny(bw2, edge2, 50, 150, 3);

    namedWindow("bw_left", WINDOW_NORMAL);
    namedWindow("bw_right", WINDOW_NORMAL);
    imshow("bw_left", bw1);
    imshow("bw_right", bw2);

    namedWindow("edge_left", WINDOW_NORMAL);
    namedWindow("edge_right", WINDOW_NORMAL);
    imshow("edge_left", edge1);
    imshow("edge_right", edge2);

    undistort_images(edge1, edge2);
    filename1 = "undistorted_left"+to_string(SETUP)+".jpg";
    filename2 = "undistorted_right"+to_string(SETUP)+".jpg";
    undistorted_image1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    undistorted_image2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
    }

    compute_disparity(undistorted_image1, undistorted_image2);
    
    //Mat img3d = reproject3d();
    //create_pointcloud(img3d);

    waitKey(0);
}
