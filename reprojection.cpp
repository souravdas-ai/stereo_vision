#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;


//This function creates a PCL visualizer, sets the point cloud to view and returns a pointer
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

int main(int argc, char* argv[]) {
	int nThreads = omp_get_max_threads();
	cout << "Max threads possible " << nThreads << endl;
	omp_set_num_threads(nThreads);
	
	Mat frame;
	vector<Mat> disparity;
	VideoCapture video("/home/sourav/Videos/Disparity.avi");
	unsigned int nFrames=0, ctr;
	int frame_rows, frame_cols;
	
	//Calculating the number of frames for parallel for
	while(video.isOpened()) {
		video >> frame;
		if(frame.empty())
			break;
		else {
			nFrames++;
			disparity.push_back(frame);
			frame_rows = frame.rows;
			frame_cols = frame.cols;
			cout << "Reading frame " << nFrames << endl;
		}
	}
	video.release();
	
	//Reprojecting disparity to 3D coordinates of each pixel
	cout << "Reprojecting image to 3D" << endl;
	vector<Mat> img3d;
	Mat Q;
	FileStorage fs("stereocalib.yml", FileStorage::READ);
	if(fs.isOpened())
		fs["Q"] >> Q;
	else
		cout << "Error in reading from calibration file" << endl;
	
	//Running parallel for using OpenMPI
	#pragma omp parallel for
	for(ctr=0; ctr<disparity.size(); ctr++) {
		reprojectImageTo3D(disparity.at(ctr), img3d.at(ctr), Q, false, CV_32F);
	}
	cout << "Reprojection complete" << endl;
	
	//Projecting to 3D point clouds
	cout << "Creating 3D point clouds" << endl;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
	double px, py, pz;
	uchar pr, pg, pb;
	for(int i=0; i<frame_rows; i++) {
		uchar* rgb_ptr = img_rgb.ptr<uchar>(i);
		double* recons_ptr = recons3D.ptr<double>(i);
		for(int j=0; j<frame_cols; j++) {
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
