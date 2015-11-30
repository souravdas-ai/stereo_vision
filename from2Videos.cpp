// This program reads from two video files (coming from a stereo camera) 
// and publishes the two images as ros topics. 
// TODO: Clean it up.

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

static const string OPENCV_WINDOW = "Image window";

int main(int argc, char** argv) {

  if (argc<3) {
    cout << "argc: "<<argc<<" \nProgram usage: "<<argv[0]<<" left_cam.mp4 right_cam.mp4" <<endl;
    return -1;
  }
  ros::init(argc, argv, "fromVideo");
  ros::NodeHandle nh;
  Mat frameLeft; 
  Mat frameRight;
  // TODO: Please ensure we are publishing the following topics properly
  // /stereo/left/image_raw
  // /stereo/left/camera_info
  // /stereo/right/image_raw
  // /stereo/right/camera_info

  sensor_msgs::ImagePtr msgLeft;
  sensor_msgs::ImagePtr msgRight;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pubLeftImg = it.advertise("/stereo/left/image_raw", 1);
  image_transport::Publisher pubRightImg = it.advertise("/stereo/right/image_raw", 1);

  VideoCapture capLeft(argv[1]);
  VideoCapture capRight(argv[2]);



  if(!capLeft.isOpened()) {
    cout << "Cannot open the left camera video: " <<argv[1]<< endl;
    cout << "Program usage: "<<argv[0]<<" "<<argv[1] << " "<<argv[2] <<endl;
    return -1;
  }

  if(!capRight.isOpened()) {
    cout << "Cannot open the right camera video: " <<argv[1]<< endl;
    cout << "Program usage: "<<argv[0]<<" "<<argv[1] << " "<<argv[2] <<endl;
    return -1;
  }

  namedWindow(OPENCV_WINDOW);
  ros::Rate loop_rate(30);


  while(nh.ok()) {
    // Unable to read next frame
    // so we break out of the loop.
    if(!capLeft.read(frameLeft))
      break;
    if(!capRight.read(frameRight))
      break;

    if(!frameLeft.empty()&&!frameRight.empty()) {
      // TODO: Ensure the Header has the same timestamp.
      msgLeft = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frameLeft).toImageMsg();
      msgRight = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frameRight).toImageMsg();

      pubLeftImg.publish(msgLeft);
      pubRightImg.publish(msgRight);
      
      // imshow(OPENCV_WINDOW, frame);
      // cv::waitKey(1);
    }
    ros::spinOnce();
    loop_rate.sleep();

  } // while(true)
  ros::spin();
  return 0;
}
