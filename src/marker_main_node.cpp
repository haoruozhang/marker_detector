/******************************************************************************
 Name        : marker_detector
 Author      : Haoruo Zhang
 E-mail      : haoruozhang[at]foxmail.com
 Copyright   : BSD
 Description : A ROS node for detecting the marker.
*******************************************************************************/

//ros
#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <objrecog_msgs/rgbd_image.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Vector3.h>
#include <tf/LinearMath/Transform.h>
#include <tf/tfMessage.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

//opencv
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <marker_detector/MarkerDetector.hpp>

//opengl
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

//C++
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <time.h>

#define PI 3.14159

using namespace std;
using namespace cv;

static GLint imagewidth;  
static GLint imageheight;  
static GLint pixellength;  
static GLubyte* pixeldata;  
#define GL_BGR_EXT 0x80E0  

Matrix44 projectionMatrix; 
Matrix44 glMatrix; 
vector<Marker> m_detectedMarkers;  
GLuint defaultFramebuffer, colorRenderbuffer;  

MarkerDetector markerDetector;

Mat_<float>  camMatrix;
Mat_<float>  distCoeff;

//camera_factor
const double camera_factor = 1;
const double camera_cx = 319.5;//325.5//315.5//319.5
const double camera_cy = 239.5;//253.5//233.5//239.5
const double camera_fx = 525.0;//518.0//570.3422
const double camera_fy = 525.0;//519.0

Mat rgb_image;

vector <GLdouble> vertices;
vector <GLdouble> normal_vectors;

int STLRead(string nameStl)
{
	clock_t start,finish;  
	double totaltime;  
	start=clock();  

	string line;
	vector<string> tmp;
        char filename[20];
        int len = nameStl.length();
        nameStl.copy(filename,len,0);
        *(filename+len)='\0';
	ifstream fin;
	fin.open(filename);
	if(!fin)
	{
		cout<<"read error"<<endl;
		return 0;
	}
	while(std::getline(fin,line))
	{
		char *Del=" ";
		char buff[100];
		strcpy(buff,line.c_str());
		char* token=strtok(buff,Del);
		while(token){
			tmp.push_back(token);
			token=strtok(NULL,Del);
		}
	}
	fin.close();

	for(int i=0;i<tmp.size();i++)
	{
		if(strcasecmp(tmp[i].c_str(), "normal")==0)
		{
			normal_vectors.push_back(atof(tmp[i+1].c_str()));
			normal_vectors.push_back(atof(tmp[i+2].c_str()));
			normal_vectors.push_back(atof(tmp[i+3].c_str()));
		}
		if(strcasecmp(tmp[i].c_str(), "vertex")==0)
		{
			vertices.push_back(atof(tmp[i+1].c_str())*100/6);
			vertices.push_back(atof(tmp[i+2].c_str())*100/6);
			vertices.push_back(atof(tmp[i+3].c_str())*100/6);
		}
	}
        cout << "normal_vectors: " << normal_vectors.size() << endl;
        cout << "vertices: " << vertices.size() << endl;
	cout<<"read success"<<endl;

	finish=clock();  
	totaltime=(double)(finish-start)/CLOCKS_PER_SEC;  
	cout<<"time"<<totaltime<<"s"<<endl;
	return 1;
}
Mat getopencvmat(Matrix44 matrix_4d)
{
    Mat opencv_mat = Mat::zeros(4,4,CV_32FC1);
    opencv_mat.at<float>(0,0) = matrix_4d.data[0];
    opencv_mat.at<float>(1,0) = matrix_4d.data[1];
    opencv_mat.at<float>(2,0) = matrix_4d.data[2];
    opencv_mat.at<float>(3,0) = matrix_4d.data[3];
    opencv_mat.at<float>(0,1) = matrix_4d.data[4];
    opencv_mat.at<float>(1,1) = matrix_4d.data[5];
    opencv_mat.at<float>(2,1) = matrix_4d.data[6];
    opencv_mat.at<float>(3,1) = matrix_4d.data[7];
    opencv_mat.at<float>(0,2) = matrix_4d.data[8];
    opencv_mat.at<float>(1,2) = matrix_4d.data[9];
    opencv_mat.at<float>(2,2) = matrix_4d.data[10];
    opencv_mat.at<float>(3,2) = matrix_4d.data[11];
    opencv_mat.at<float>(0,3) = matrix_4d.data[12];
    opencv_mat.at<float>(1,3) = matrix_4d.data[13];
    opencv_mat.at<float>(2,3) = matrix_4d.data[14];
    opencv_mat.at<float>(3,3) = matrix_4d.data[15];
    return opencv_mat;
}
Matrix44 getMatrix44(Mat opencv_mat)
{
    Matrix44 matrix_4d;
    matrix_4d.data[0] = opencv_mat.at<float>(0,0);
    matrix_4d.data[1] = opencv_mat.at<float>(1,0);
    matrix_4d.data[2] = opencv_mat.at<float>(2,0);
    matrix_4d.data[3] = opencv_mat.at<float>(3,0);
    matrix_4d.data[4] = opencv_mat.at<float>(0,1);
    matrix_4d.data[5] = opencv_mat.at<float>(1,1);
    matrix_4d.data[6] = opencv_mat.at<float>(2,1);
    matrix_4d.data[7] = opencv_mat.at<float>(3,1);
    matrix_4d.data[8] = opencv_mat.at<float>(0,2);
    matrix_4d.data[9] = opencv_mat.at<float>(1,2);
    matrix_4d.data[10] = opencv_mat.at<float>(2,2);
    matrix_4d.data[11] = opencv_mat.at<float>(3,2);
    matrix_4d.data[12] = opencv_mat.at<float>(0,3);
    matrix_4d.data[13] = opencv_mat.at<float>(1,3);
    matrix_4d.data[14] = opencv_mat.at<float>(2,3);
    matrix_4d.data[15] = opencv_mat.at<float>(3,3);
    return matrix_4d;
}
void readCameraParameter1()
{
    //calibratoin data for kinect
    camMatrix = Mat::eye(3, 3, CV_64F);
    distCoeff = Mat::zeros(8, 1, CV_64F); 

    camMatrix(0,0) = camera_fx;  
    camMatrix(1,1) = camera_fy;  
    camMatrix(0,2) = camera_cx; //640  
    camMatrix(1,2) = camera_cy; //480,?????????  
  
    for (int i=0; i<4; i++)  
        distCoeff(i,0) = 0;
}
void build_projection(Mat_<float> cameraMatrix)  
{  
    float near = 0.01;  // Near clipping distance  
    float far = 100;  // Far clipping distance  

    float f_x = cameraMatrix(0,0); // Focal length in x axis  
    float f_y = cameraMatrix(1,1); // Focal length in y axis (usually the same?)  
    float c_x = cameraMatrix(0,2); // Camera primary point x  
    float c_y = cameraMatrix(1,2); // Camera primary point y  
  
    projectionMatrix.data[0] =  - 2.0 * f_x / imagewidth;  
    projectionMatrix.data[1] = 0.0;  
    projectionMatrix.data[2] = 0.0;  
    projectionMatrix.data[3] = 0.0;  
  
    projectionMatrix.data[4] = 0.0;  
    projectionMatrix.data[5] = 2.0 * f_y / imageheight;  
    projectionMatrix.data[6] = 0.0;  
    projectionMatrix.data[7] = 0.0;  
  
    projectionMatrix.data[8] = 2.0 * c_x / imagewidth - 1.0;  
    projectionMatrix.data[9] = 2.0 * c_y / imageheight - 1.0;      
    projectionMatrix.data[10] = -( far+near ) / ( far - near );  
    projectionMatrix.data[11] = -1.0;  
  
    projectionMatrix.data[12] = 0.0;  
    projectionMatrix.data[13] = 0.0;  
    projectionMatrix.data[14] = -2.0 * far * near / ( far - near );          
    projectionMatrix.data[15] = 0.0;  
}  
void setMarker(const vector<Marker>& detectedMarkers)  
{  
    m_detectedMarkers = detectedMarkers;  
}  
void LightInit2(void)
{
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);
	GLfloat ambientLight[] = {0.1f,0.1f,0.1f,1.0f};
	GLfloat diffuseLight[]={1.0f,1.0f,1.0f,1.0f};
	GLfloat specular[]={1.0f,1.0f,1.0f,1.0f};
	GLfloat lightPos[]={100.0f,100.0f,100.0f,0.0f};
	GLfloat gray[]={0.75f,1.0f,0.75f,1.0f};
	glEnable(GL_LIGHTING);
	glLightfv(GL_LIGHT0,GL_AMBIENT,ambientLight);
	glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuseLight);
	glLightfv(GL_LIGHT0,GL_SPECULAR,specular);
	glLightfv(GL_LIGHT0,GL_POSITION,lightPos);
	glEnable(GL_LIGHT0);
	//glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);
	//glMaterialfv(GL_FRONT,GL_SPECULAR,specular);
	//glMateriali(GL_FRONT,GL_SHININESS,128);
	//glClearColor(0.0f,0.0f,0.0f,1.0f);
	//glColor4f(0.75,0.75,0.75,1.0);
	glShadeModel(GL_FLAT);
} 
void display(void)  
{  
      
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);   
    glDrawPixels(imagewidth,imageheight,GL_BGR_EXT,GL_UNSIGNED_BYTE,pixeldata);  
    
    glMatrixMode(GL_PROJECTION);  
    glLoadMatrixf(projectionMatrix.data);  
    glMatrixMode(GL_MODELVIEW);  
    glLoadIdentity();  
    
    glEnableClientState(GL_VERTEX_ARRAY);  //??????????
    glEnableClientState(GL_NORMAL_ARRAY);  
  
    glPushMatrix();  
    glLineWidth(3.0f);  
  
    Mat transform_d = Mat::ones(4,4,CV_32FC1);
    transform_d.at<float>(0,3) = 5.0/3;
    transform_d.at<float>(1,3) = 5.0/3;


   for (int i=0;i<16;i++) 
   { glMatrix.data[i]=0;}
   for (size_t transformationIndex=0; transformationIndex<m_detectedMarkers.size(); transformationIndex++)  
    {  
        Transformation& transformation = m_detectedMarkers[transformationIndex].transformation; 
        int marker_id = m_detectedMarkers[transformationIndex].id;
        cout << "ID: " << marker_id << endl;
        
        if (marker_id == 213)
        {
            glMatrix = transformation.getMat44();
        }
   }
   glLoadMatrixf(reinterpret_cast<const GLfloat*>(&glMatrix.data[0]));
  
   glBegin(GL_LINES);
   glColor4f(0.0f, 1.0f, 0.0f, 0.0f);
   glVertex3f(0.0,0.0,0.0);
   glVertex3f(2,0.0,0.0);
   glColor4f(1.0f, 0.0f, 0.0f, 0.0f);
   glVertex3f(0.0,0.0,0.0);
   glVertex3f(0.0,2,0.0);
   glColor4f(0.0f, 0.0f, 1.0f, 0.0f);
   glVertex3f(0.0,0.0,0.0);
   glVertex3f(0.0,0.0,2);
   glEnd();

   glPopMatrix();
   glDisableClientState(GL_VERTEX_ARRAY);
   glutSwapBuffers();
}  
int show(const char* filename,int argc, char** argv,Mat_<float>& cameraMatrix, vector<Marker>& detectedMarkers)  
{  
    FILE* pfile=fopen(filename,"rb");  
    if(pfile == 0) exit(0);  
    fseek(pfile,0x0012,SEEK_SET);  
    fread(&imagewidth,sizeof(imagewidth),1,pfile);  
    fread(&imageheight,sizeof(imageheight),1,pfile);  
    pixellength=imagewidth*3;  
    while(pixellength%4 != 0)pixellength++;  
    pixellength *= imageheight;  
    pixeldata = (GLubyte*)malloc(pixellength);  
    if(pixeldata == 0) exit(0);  
    fseek(pfile,54,SEEK_SET);  
    fread(pixeldata,pixellength,1,pfile);  
    fclose(pfile); 
    build_projection(cameraMatrix);
    setMarker(detectedMarkers);
    free(pixeldata);  
    return 0;  
}

tf::Transform getcorrecttf(Matrix44 matrix_4d)
{
     tf::Vector3 origin;
     origin.setValue(matrix_4d.data[13]*6/100, matrix_4d.data[12]*6/100, matrix_4d.data[14]*6/100); //size 6cm
     tf::Matrix3x3 tf3d;
     tf3d.setValue(matrix_4d.data[5], matrix_4d.data[1], matrix_4d.data[9], 
                   matrix_4d.data[4], matrix_4d.data[0], matrix_4d.data[8], 
                   matrix_4d.data[6], matrix_4d.data[2], matrix_4d.data[10]);
     tf::Matrix3x3 tf3d_2;
     tf3d_2.setValue(0, -1, 0, 
                     -1, 0, 0, 
                     0, 0, -1);
     tf::Quaternion tfqt;
     tf3d.getRotation(tfqt);

     tf::Quaternion tfqt_2;
     tf3d_2.getRotation(tfqt_2);
     tf::Transform transform_d;
     transform_d.setOrigin( origin );
     transform_d.setRotation( tfqt );
     tf::Transform transform_c;

     transform_c.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
     transform_c.setRotation( tfqt_2 );

     transform_d = transform_c*transform_d;
     return transform_d;
}
int main(int argc, char** argv)
{
      ros::init(argc, argv, "marker_main_node");
      ros::NodeHandle nh;
      ros::ServiceClient client = nh.serviceClient<objrecog_msgs::rgbd_image>("get_image");
      objrecog_msgs::rgbd_image srv;
      srv.request.start = true;
      sensor_msgs::Image msg_rgb;

      tf::TransformBroadcaster br;
      tf::Transform transform_b;

      transform_b.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
      transform_b.setRotation( tf::Quaternion(0, 0, 0, 1) );
      br.sendTransform(tf::StampedTransform(transform_b, ros::Time::now(), "camera_rgb_optical_frame", "marker_frame"));//initiate tf

      vector<Marker> markers;
      readCameraParameter1();
      cout << camMatrix << endl;

      imagewidth = 640;
      imageheight = 480;
      glutInit(&argc,argv);  
      glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);  
      glutInitWindowPosition(100,100);  
      glutInitWindowSize(imagewidth,imageheight);
      glutCreateWindow("test.bmp");
      LightInit2(); 
      glutDisplayFunc(&display);
      glutLeaveMainLoop();

      build_projection(camMatrix);
      setMarker(markers);

      ros::Rate loop_rate(20);
      while (ros::ok())
      {  
          if (client.call(srv))
          {
            try
            {         
              msg_rgb = srv.response.rgb_image;
              rgb_image = cv_bridge::toCvCopy(msg_rgb, sensor_msgs::image_encodings::TYPE_8UC3)->image; 
            }
            catch (cv_bridge::Exception& e)
            {
              ROS_ERROR("cv_bridge exception: %s", e.what());
              return 1;
            }
            IplImage ipl_rgb_image = rgb_image;
            cvConvertImage(&ipl_rgb_image , &ipl_rgb_image , CV_CVTIMG_SWAP_RB);

            markerDetector.processFrame(rgb_image,camMatrix, distCoeff,markers);
            string testXml = ros::package::getPath("marker_detector") + "/test.bmp";
            imwrite(testXml,rgb_image);          
            show(testXml.data(),argc,argv,camMatrix, markers);
            display();

            transform_b = getcorrecttf(glMatrix);


             br.sendTransform(tf::StampedTransform(transform_b, ros::Time::now(), "camera_rgb_optical_frame", "marker_frame"));
          }
          else
          {
            ROS_ERROR("Failed to call service get_image");
          }
       ros::spinOnce();
       loop_rate.sleep();
    }
    ros::spin();
    return 0;
}


