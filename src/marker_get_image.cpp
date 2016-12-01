/******************************************************************************
 Name        : marker_get_image
 Author      : Haoruo Zhang
 E-mail      : haoruozhang[at]foxmail.com
 Copyright   : BSD
 Description : A ROS node for getting rgbd image.
*******************************************************************************/

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <rgbd_srvs/rgbd_image.h>

using namespace std;
using namespace cv;

sensor_msgs::Image rgb_image;
sensor_msgs::Image depth_image;

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_rgb;
  image_transport::Subscriber image_sub_depth;
  
public:

  ImageConverter()
    : it_(nh_)
  {
    image_sub_rgb = it_.subscribe("/camera/rgb/image_rect_color", 1, &ImageConverter::imageCb_rgb, this);
    image_sub_depth = it_.subscribe("/camera/depth_registered/image_raw", 1, &ImageConverter::imageCb_depth, this);
  }

  ~ImageConverter()
  {
  }

  void imageCb_rgb(const sensor_msgs::ImageConstPtr& msg)
  {
     rgb_image = *msg;
  }
  void imageCb_depth(const sensor_msgs::ImageConstPtr& msg)
  {  
     depth_image = *msg;
  }

};

bool marker_get_image(rgbd_srvs::rgbd_image::Request &req,rgbd_srvs::rgbd_image::Response &res)
{
    if (req.start) 
    {
         res.rgb_image = rgb_image;
         res.depth_image = depth_image;
    }
    ROS_INFO("success");
    return true;
}


int main(int argc, char** argv)
{    
   
  ros::init(argc, argv, "marker_get_image");
  ImageConverter ic; 
  ros::NodeHandle n;
  ros::ServiceServer service = n.advertiseService("marker_get_image", marker_get_image);

  ros::Rate loop_rate(200);
  while (ros::ok())
  {

     ros::spinOnce();
    loop_rate.sleep();
  }
  ros::spin();
  return 0;
}

