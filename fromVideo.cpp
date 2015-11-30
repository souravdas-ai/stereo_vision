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
  ros::init(argc, argv, "fromVideo");
  ros::NodeHandle nh;
  Mat frame; 
  //fromVideo ic;
  sensor_msgs::ImagePtr msg;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("camera/image", 1);
  VideoCapture cap(argv[1]);
  if(!cap.isOpened()) {
    cout << "Cannot open the video: " <<argv[1]<< endl;
    cout << "Program usage: "<<argv[0]<<" "<<argv[1] << endl;
    return -1;
  }
  namedWindow(OPENCV_WINDOW);
  ros::Rate loop_rate(30);


  while(nh.ok()) {
    // Unable to read next frame
    // so we break out of the loop.
    if(!cap.read(frame))
      break;
    if(!frame.empty()) {
      msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
      pub.publish(msg);
      
      // imshow(OPENCV_WINDOW, frame);
      // cv::waitKey(1);
    }
    ros::spinOnce();
    loop_rate.sleep();

  } // while(true)
  ros::spin();
  return 0;
}
