#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>       //stringstream

#include <marker_detector/MarkerDetector.hpp>

using namespace cv;
using namespace std;

float perimeter(const vector<Point2f> &a)//求多边形周长。
{
  float sum=0,dx,dy;
  for(size_t i=0;i<a.size();i++)
    {
      size_t i2=(i+1) % a.size();

      dx = a[i].x - a[i2].x;
      dy = a[i].y - a[i2].y;

      sum += sqrt(dx*dx + dy*dy);
    }

  return sum;
}

MarkerDetector::MarkerDetector()
     :m_minContourLengthAllowed(100.0)
     ,markerSize(100,100)
 {
   bool centerOrigin = true;
   if(centerOrigin)
     {
       m_markerCorners3d.push_back(Point3f(-0.5f,-0.5f,0));
       m_markerCorners3d.push_back(Point3f(+0.5f,-0.5f,0));
       m_markerCorners3d.push_back(Point3f(+0.5f,+0.5f,0));
       m_markerCorners3d.push_back(Point3f(-0.5f,+0.5f,0));
     }
   else
     {
       m_markerCorners3d.push_back(Point3f(0,0,0));
       m_markerCorners3d.push_back(Point3f(1,0,0));
       m_markerCorners3d.push_back(Point3f(1,1,0));
       m_markerCorners3d.push_back(Point3f(0,1,0));
     }

   m_markerCorners2d.push_back(Point2f(0,0));
   m_markerCorners2d.push_back(Point2f(markerSize.width-1,0));
   m_markerCorners2d.push_back(Point2f(markerSize.width-1,markerSize.height-1));
   m_markerCorners2d.push_back(Point2f(0,markerSize.height-1));

}

const vector<Transformation>& MarkerDetector::getTransformations() const
{
  return m_transformations;
}

void MarkerDetector::prepareImage(const Mat& src,Mat& grayscale)
 {
  //彩色转换成灰色图像
  cvtColor(src,grayscale,CV_BGRA2GRAY);
}

//绝对阈值结果取决于光照条件和软强度变化。采用自适应阈值法，以像素为单位，将给定半径内的所有像素的平均强度作为该像素的强度，使接下来的轮廓检测更具有鲁棒性。
void MarkerDetector::performThreshold(const Mat& grayscale,Mat& thresholdImg)
{
  /*输入图像  
  //输出图像  
  //使用 CV_THRESH_BINARY 和 CV_THRESH_BINARY_INV 的最大值  
  //自适应阈值算法使用：CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C   
  //取阈值类型：必须是下者之一  
  //CV_THRESH_BINARY,  
  //CV_THRESH_BINARY_INV  
  //用来计算阈值的象素邻域大小: 3, 5, 7, ...  
  */  
  adaptiveThreshold(grayscale,//Input Image
                    thresholdImg,//Result binary image
                    255,
                    ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY_INV,
                    7,
                    7
                    );
#ifdef SHOW_DEBUG_IMAGES
  imshow("Threshold image",thresholdImg);
  imwrite("Threshold image" + ".png",thresholdImg);    
#endif
}
//检测所输入的二值图像的轮廓，返回一个多边形列表，其每个多边形标识一个轮廓，小轮廓不关注，不包括标记...
void MarkerDetector::findContour(cv::Mat& thresholdImg, ContoursVector& contours, int minContourPointsAllowed) const
{
  ContoursVector allContours;
  /*输入图像image必须为一个2值单通道图像  
  //检测的轮廓数组，每一个轮廓用一个point类型的vector表示  
  //轮廓的检索模式  
   
     CV_RETR_EXTERNAL表示只检测外轮廓 
     CV_RETR_LIST检测的轮廓不建立等级关系 
     CV_RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。 
     CV_RETR_TREE建立一个等级树结构的轮廓。具体参考contours.c这个demo 
       
  //轮廓的近似办法  
  
     CV_CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1 
     CV_CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息 
     CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法 
     offset表示代表轮廓点的偏移量，可以设置为任意值。对ROI图像中找出的轮廓，并要在整个图像中进行分析时，这个参数还是很有用的。 
   */  
  findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  contours.clear();
  for(size_t i=0;i<allContours.size();i++)
    {
      int contourSize = allContours[i].size();
      if(contourSize > minContourPointsAllowed)
        {
          contours.push_back(allContours[i]);
        }
    }
  //Mat result(src.size(),CV_8U,Scalar(0)); 
  //drawContours(result,detectedMarkers,-1,Scalar(255),2); 
  //imshow("AR based marker...",result);
#ifdef SHOW_DEBUG_IMAGES
    {
      Mat contoursImage(thresholdImg.size(), CV_8UC1);
      contoursImage = Scalar(0);
      drawContours(contoursImage, contours, -1, cv::Scalar(255), 2, CV_AA);
      imshow("Contours",contoursImage);
      imwrite("Contours" + ".png",contoursImage);            
    }   
#endif
}

//由于我们的标记是四边形，当找到图像所有轮廓细节后，本文用Opencv内置API检测多边形，通过判断多边形定点数量是否为4，四边形各顶点之间相互距离是否满足要求(四边形是否足够大)，过滤非候选区域。然后再根据候选区域之间距离进一步筛选，得到最终的候选区域，并使得候选区域的顶点坐标逆时针排列。
void MarkerDetector::findCandidates(const ContoursVector& contours,vector<Marker>& detectedMarkers)
{
  vector<Point> approxCurve;//返回结果为多边形，用点集表示//相似形状
  vector<Marker> possibleMarkers;//可能的标记

  //For each contour,分析它是不是像标识，找到候选者//分析每个标记，如果是一个类似标记的平行六面体...
  for(size_t i=0;i<contours.size();i++)
    {
      /*近似一个多边形逼近，为了减少轮廓的像素。这样比较好，可筛选出非标记区域，因为标记总能被四个顶点的多边形表示。如果多边形的顶点多于或少于四个，就绝对不是本项目想要的标记。通过点集近似多边形，第三个参数为epsilon代表近似程度，即原始轮廓及近似多边形之间的距离，第四个参数表示多边形是闭合的。*/
      double eps = contours[i].size()*0.05;
      //输入图像的2维点集，输出结果，估计精度，是否闭合。输出多边形的顶点组成的点集//使多边形边缘平滑，得到近似的多边形 
      approxPolyDP(contours[i],approxCurve,eps,true);

      //我们感兴趣的多边形只有四个顶点
      if(approxCurve.size() != 4)
        continue;

      //检查轮廓是否是凸边形
      if(!isContourConvex(approxCurve))
        continue;

      //确保连续点之间的距离是足够大的。//确保相邻的两点间的距离“足够大”－大到是一条边而不是短线段就是了
      //float minDist = numeric_limits<float>::max();//代表float可以表示的最大值，numeric_limits就是模板类，这里表示max（float）;3.4e038
      float minDist = 1e10;//这个值就很大了

      //求当前四边形各顶点之间的最短距离
      for(int i=0;i<4;i++)
        {
          Point side = approxCurve[i] - approxCurve[(i+1)%4];//这里应该是2维的相减
          float squaredSideLength = side.dot(side);//求2维向量的点积，就是XxY
          minDist = min(minDist,squaredSideLength);//找出最小的距离
        }

      //检查距离是不是特别小，小的话就退出本次循环，开始下一次循环
      if(minDist<m_minContourLengthAllowed)
        continue;

      //所有的测试通过了，保存标识候选，当四边形大小合适，则将该四边形maker放入possibleMarkers容器内 //保存相似的标记   
      Marker m;
      for(int i=0;i<4;i++)
        m.points.push_back(Point2f(approxCurve[i].x,approxCurve[i].y));//vector头文件里面就有这个push_back函数，在vector类中作用为在vector尾部加入一个数据。

      /*逆时针保存这些点
      //从代码推测，marker中的点集本来就两种序列：顺时针和逆时针，这里要把顺时针的序列改成逆时针，在多边形逼近时，多边形是闭合的，则不是顺时针就是逆时针
      //在第一个和第二个点之间跟踪出一条线，如果第三个点在右边，则点是逆时针保存的//逆时针排列这些点,第一个点和第二个点之间连一条线,如果第三个点在边，那么这些点就是逆时针*/
      Point v1 = m.points[1] - m.points[0];
      Point v2 = m.points[2] - m.points[0];

      /*行列式的几何意义是什么呢？有两个解释：一个解释是行列式就是行列式中的行或列向量所构成的超平行多面体的有向面积或有向体积；另一个解释是矩阵A的行列式detA就是线性变换A下的图形面积或体积的伸缩因子。
      //以行向量a=(a1,a2)，b=(b1,b2)为邻边的平行四边形的有向面积：若这个平行四边形是由向量沿逆时针方向转到b而得到的，面积取正值；若这个平行四边形是由向量a沿顺时针方向转到而得到的，面积取负值； */
      double o = (v1.x * v2.y) - (v1.y * v2.x);

      if(o<0.0) //如果第三个点在左边，那么交换第一个点和第三个点，逆时针保存
        swap(m.points[1],m.points[3]);

      possibleMarkers.push_back(m);//把这个标识放入候选标识向量中
    }

  //移除那些角点互相离的太近的四边形//移除角点太接近的元素  
  vector< pair<int,int> > tooNearCandidates;
  for(size_t i=0;i<possibleMarkers.size();i++)
    {
      const Marker& m1 = possibleMarkers[i];
      //计算两个maker四边形之间的距离，四组点之间距离和的平均值，若平均值较小，则认为两个maker很相近,把这一对四边形放入移除队列。//计算每个边角到其他可能标记的最近边角的平均距离
      for(size_t j=i+1;j<possibleMarkers.size();j++)
        {
          const Marker& m2 = possibleMarkers[j];
          float distSquared = 0;
          for(int c=0;c<4;c++)
            {
              Point v = m1.points[c] - m2.points[c];
              //向量的点乘－》两点的距离
              distSquared += v.dot(v);
            }
          distSquared /= 4;

          if(distSquared < 100)
            {
              tooNearCandidates.push_back(pair<int,int>(i,j));
            }
        }
    }

  //移除了相邻的元素对的标识
  //计算距离相近的两个marker内部，四个点的距离和，将距离和较小的，在removlaMask内做标记，即不作为最终的detectedMarkers 
  vector<bool> removalMask(possibleMarkers.size(),false);//创建Vector对象，并设置容量。第一个参数是容量，第二个是元素。

  for(size_t i=0;i<tooNearCandidates.size();i++)
    {
      //求这一对相邻四边形的周长
      float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
      float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);

      //谁周长小，移除谁
      size_t removalIndex;
      if(p1 > p2)
        removalIndex = tooNearCandidates[i].second;
      else
        removalIndex = tooNearCandidates[i].first;

      removalMask[removalIndex] = true;
    }

  //返回候选，移除相邻四边形中周长较小的那个，放入待检测的四边形的队列中。//返回可能的对象
  detectedMarkers.clear();
  for(size_t i = 0;i<possibleMarkers.size();i++)
    {
      if(!removalMask[i])
        detectedMarkers.push_back(possibleMarkers[i]);
    }

}

void MarkerDetector::recognizeMarkers(const Mat& grayscale,vector<Marker>& detectedMarkers)
{
  Mat canonicalMarkerImage;
  char name[20] = "";

  vector<Marker> goodMarkers;

  /*Identify the markers识别标识 //分析每一个捕获到的标记，去掉透视投影，得到平面／正面的矩形。
  //为了得到这些矩形的标记图像，我们不得不使用透视变换去恢复(unwarp)输入的图像。这个矩阵应该使用cv::getPerspectiveTransform函数，它首先根据四个对应的点找到透视变换，第一个参数是标记的坐标，第二个是正方形标记图像的坐标。估算的变换将会把标记转换成方形，从而方便我们分析。 */
  for(size_t i=0;i<detectedMarkers.size();i++)
    {
      Marker& marker = detectedMarkers[i];
      //找到透视转换矩阵，获得矩形区域的正面视图// 找到透视投影，并把标记转换成矩形，输入图像四边形顶点坐标，输出图像的相应的四边形顶点坐标 
      // Find the perspective transformation that brings current marker to rectangular form
      Mat markerTransform = getPerspectiveTransform(marker.points,m_markerCorners2d);//输入原始图像和变换之后的图像的对应4个点，便可以得到变换矩阵
      /* Transform image to get a canonical marker image
      // Transform image to get a canonical marker image  
      //输入的图像  
      //输出的图像  
      //3x3变换矩阵 */
      warpPerspective(grayscale,canonicalMarkerImage,markerTransform,markerSize);//对图像进行透视变换,这就得到和标识图像一致正面的图像，方向可能不同，看四个点如何排列的了。感觉这个变换后，就得到只有标识图的正面图

      // sprintf(name,"warp_%d.jpg",i);
      // imwrite(name,canonicalMarkerImage);
#ifdef SHOW_DEBUG_IMAGES
         {
          Mat markerImage = grayscale.clone();
          marker.drawContour(markerImage);
          Mat markerSubImage = markerImage(boundingRect(marker.points));

          imshow("Source marker" + ToString(i),markerSubImage);
          imwrite("Source marker" + ToString(i) + ".png",markerSubImage);   
          
          imshow("Marker " + ToString(i),canonicalMarkerImage);
          imwrite("Marker " + ToString(i) + ".png",canonicalMarkerImage);   
        }
#endif

      int nRotations;
      int id = Marker::getMarkerId(canonicalMarkerImage,nRotations);
     // cout << "ID: " << id << endl;
      
      if(id!=-1)
        {
          marker.id = id;
          //sort the points so that they are always in the same order no matter the camera orientation  
          //Rotates the order of the elements in the range [first,last), in such a way that the element pointed by middle becomes the new first element.
          //根据相机的旋转，调整标记的姿态
          rotate(marker.points.begin(),marker.points.begin() + 4 - nRotations,marker.points.end());//就是一个循环移位

          goodMarkers.push_back(marker);
        }
    }

  //refine using subpixel accuracy the corners  是把所有标识的四个顶点都放在一个大的向量中。
  if(goodMarkers.size() > 0)
    {
      //找到所有标记的角点 
      vector<Point2f> preciseCorners(4*goodMarkers.size());//每个marker四个点
      for(size_t i=0;i<goodMarkers.size();i++)
        {
          Marker& marker = goodMarkers[i];

          for(int c=0;c<4;c++)
             {
              preciseCorners[i*4+c] = marker.points[c];//i表示第几个marker，c表示某个marker的第几个点
            }
        }

      //Refines the corner locations.The function iterates to find the sub-pixel accurate location of corners or radial saddle points
      //类型  
      /*  
         CV_TERMCRIT_ITER 用最大迭代次数作为终止条件 
         CV_TERMCRIT_EPS 用精度作为迭代条件 
         CV_TERMCRIT_ITER+CV_TERMCRIT_EPS 用最大迭代次数或者精度作为迭代条件，决定于哪个条件先满足  
      //迭代的最大次数  
      //特定的阀值 */
      TermCriteria termCriteria = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,30,0.01);//这个是迭代终止条件，这里是达到30次迭代或者达到0.01精度终止。角点精准化迭代过程的终止条件
      /*输入图像  
      //输入的角点，也作为输出更精确的角点  
      //接近的大小（Neighborhood size）  
      //Aperture parameter for the Sobel() operator  
      //像素迭代（扩张）的方法 */
      cornerSubPix(grayscale,preciseCorners,cvSize(5,5),cvSize(-1,-1),termCriteria);//发现亚像素精度的角点位置，第二个参数代表输入的角点的初始位置并输出精准化的坐标。在标记检测的早期的阶段没有使用cornerSubPix函数是因为它的复杂性－调用这个函数处理大量顶点时会耗费大量的处理时间，因此我们只在处理有效标记时使用。

      //copy back，再把精准化的坐标传给每一个标识。// 保存最新的顶点
      for(size_t i=0;i<goodMarkers.size();i++)
         {
          Marker& marker = goodMarkers[i];
            for(int c=0;c<4;c++)
               {
                marker.points[c] = preciseCorners[i*4+c];
                //cout<<"X:"<<marker.points[c].x<<"Y:"<<marker.points[c].y<<endl;
              }
        }

    }

  //画出细化后的矩形图片
   Mat markerCornersMat(grayscale.size(),grayscale.type());
    markerCornersMat = Scalar(0);
    for(size_t i=0;i<goodMarkers.size();i++)
    {
      goodMarkers[i].drawContour(markerCornersMat,Scalar(255));
    }

    //imshow("Markers refined edges",grayscale*0.5 + markerCornersMat);
    //imwrite("Markers refined edges" + ".png",grayscale*0.5 + markerCornersMat);   
    //imwrite("refine.jpg",grayscale*0.5 + markerCornersMat);

  detectedMarkers = goodMarkers;

}

//对每一个标记，求出其相对于相机的转换矩阵。找到上面监测到的标记的位置
void MarkerDetector::estimatePosition(vector<Marker>& detectedMarkers,Mat_<float>& camMatrix,Mat_<float>& distCoeff)
{
  for(size_t i=0;i<detectedMarkers.size();i++)
    {
      Marker& m = detectedMarkers[i];
      
      Mat Rvec;
      Mat_<float> Tvec;//Mat_<float>对应的是CV_32F
      Mat raux,taux;
 
      //寻找物体投影的三维坐标和二维坐标之间的对应关系,利用已知的平面四点坐标确定摄像头相对世界坐标系的平移和旋转。在物体深度变化相对于物体到摄像机的距离比较大的时候,这个函数比较好。输出旋转向量和位移向量。
      /*solvePnP( InputArray _opoints, InputArray _ipoints,  
                  InputArray _cameraMatrix, InputArray _distCoeffs,  
                  OutputArray _rvec, OutputArray _tvec, bool useExtrinsicGuess ) 
      _opoints是对象坐标系的对象点集数组，应该是std::vector<cv::Point3f>对象，此处我们传标记的3维坐标系（4个点的集合）。
      _ipoints是对象所对应的图像点（投影）数组。参数应该是std::vector<cv::Point2f> 或者 cv::Mat－ 2 x N ／ N x 2,其中N是点的数量，这里我们传递我们发现的标记的角点。
      _cameraMatrix:是相机的内参矩阵。
      _distCoeffs:这是输入4 x 1,1×4、5 x 1或1 x 5向量的畸变系数(k1,k2,p1,p2,[k3])。如果它是空的,所有的畸变系数设置为0。
      _rvec是输出把点从模型坐标系转换到相机坐标系的旋转向量。
      _tvec同上，这里是输出平移向量。
      useExtrinsicGuess如果是true,那么这个函数就会使用_rvec和_tvec分别作为初始的近似旋转和平移向量，然后再进一步优化。

      我们用这个函数去计算相机转换将最大限度减少投影误差，也就是观察到的投影和预计的投影之间的距离平方和。
      估计的转换是由旋转（rotation）(_rvec)和转换（translation）组件(_tvec)构成。这也就是所谓的欧氏变换或刚性变换。
      刚性变换被定义为当一个装换作用在任何向量v，产生转换向量T(v)的形式:
      T(v) = R v + t
      RT=R-1(即R是一个正交变换),t是原始转换的向量,一个刚性转换满足：
      det(R) = 1
      这意味着R不产生反射,因此它代表一个旋转(一个保持定向正交变换)。
      为了获得一个3×3旋转矩阵的旋转向量,我们将使用cv::Rodrigues 。该函数通过旋转的向量转换一个旋转参数并返回其等效旋转矢量旋转矩阵。
      注：因为上面的solvePnP函数找到了相机相对于3维空间中的标记的位置，因此我们必须转换我们的成果。因此我们将得到的转换将在相机坐标系中描述标记的转换，这显然对渲染引擎更加友好。*/
      //solvePnP(m_markerCorners3d,m.points,camMatrix,distCoeff,raux,taux);
      solvePnP(m_markerCorners3d,m.points,camMatrix,distCoeff,raux,taux);
      raux.convertTo(Rvec,CV_32F);//转换Mat的保存类型，输出Rvec
      taux.convertTo(Tvec,CV_32F);

      Mat_<float> rotMat(3,3);
      Rodrigues(Rvec,rotMat);//罗德里格斯变换对旋转向量和旋转矩阵进行转换，输出旋转矩阵

      //Copy to transformation matrix，复制旋转矩阵到标识的类变量中
      for(int col=0;col<3;col++)
          {
          for(int row=0;row<3;row++)
              {
              m.transformation.r().mat[row][col] = rotMat(row,col);//copy rotation component
            }
          m.transformation.t().data[col] = Tvec(col);//copy translation component//复制位移向量到标识类的变量
        }

      //since solvePnP finds camera location,w.r.t to marker pose,to get marker pose w.r.t to the camera we invert it.
      //slovePnP得到了相机相对于标识的旋转矩阵和位移向量，下面将得到标识相对于相机的旋转矩阵和位移向量。反转这个矩阵
      m.transformation = m.transformation.getInverted();
    }
}

//此时，你已经知道如何去发现图像上的标记并计算它们在空间中相对于相机的准确位置。是时候去画点东西了。就像刚才说的，我们将会使用opengl函数渲染场景，3维虚拟化是增强现实核心的部分，opengl提供了所有基本的条件去创建高质量的渲染。

bool MarkerDetector::findMarkers(const Mat& frame,vector<Marker>& detectedMarkers)
{
  //Mat bgraMat(frame.height,frame.width,CV_8UC4,frame.data,frame.stride);
  prepareImage(frame,m_grayscaleImage);
  performThreshold(m_grayscaleImage,m_thresholdImg);

  findContour(m_thresholdImg,m_contours,m_grayscaleImage.cols/5);

  findCandidates(m_contours,detectedMarkers);

  recognizeMarkers(m_grayscaleImage,detectedMarkers);

  
  return false;
}

void MarkerDetector::processFrame(const Mat& frame,Mat_<float>& camMatrix,Mat_<float>& distCoeff,vector<Marker>& markers)
{
  //vector<Marker> markers;
  findMarkers(frame,markers);

  estimatePosition(markers,camMatrix,distCoeff);
  sort(markers.begin(),markers.end());

  m_transformations.clear();
  for(size_t i=0;i<markers.size();i++)
    {
      m_transformations.push_back(markers[i].transformation);
      //cout << markers[i].transformation.r().mat[0][0] << endl;
    }
}



