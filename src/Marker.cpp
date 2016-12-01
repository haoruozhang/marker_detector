#include <marker_detector/Marker.hpp>

using namespace cv;
using namespace std;

Marker::Marker()
     :id(-1)
 {

}

bool operator<(const Marker& M1,const Marker& M2)
 {
  return M1.id<M2.id;
}

Mat Marker::rotate(Mat in)//就是把矩阵旋转90度
  {
  Mat out;
  in.copyTo(out);
  for(int i=0;i<in.rows;i++)
      {
      for(int j=0;j<in.cols;j++)
          {
          out.at<uchar>(i,j)=in.at<uchar>(in.cols-j-1,i);//at<uchar>用来指定某个位置的像素，同时指定数据类型。就是交换元素，怎么交换的？
        }
    }
  return out;
}

//在信息论中，两个等长字符串之间的汉明距离是两个字符串对应位置的字符不同的个数。换句话说，它就是将一个字符串变换成另外一个字符串所需要替换的字符个数。
int Marker::hammDistMarker(Mat bits)//对每个可能的标记方向找到海明距离，和参考标识一致的为0,其他旋转形式的标记不为0,因为经过透射变换后，只能得到四个方向的标记，则旋转四次，找到和参考标识一致的方向。
 {
  int ids[4][5]=
     {
        {1,0,0,0,0},
        {1,0,1,1,1},
        {0,1,0,0,1},
        {0,1,1,1,0}
    };

  int dist = 0;

  for(int y=0;y<5;y++)
     {
      int minSum = 1e5;//每个元素的海明距离

      for(int p=0;p<4;p++)
         {
          int sum=0;
          //now,count
          for(int x=0;x<5;x++)
             {
              sum += bits.at<uchar>(y,x) == ids[p][x]?0:1;
            }
          if(minSum>sum)
            minSum=sum;
        }

      dist += minSum;
    }

  return dist;
}

int Marker::mat2id(const Mat& bits)//移位，求或，再移位，得到最终的ID
 {
  int val=0;
  for(int y=0;y<5;y++)
     {
      val<<=1;//移位操作
      if(bits.at<uchar>(y,1)) val |= 1;
      val<<=1;
      if(bits.at<uchar>(y,3)) val |= 1;
    }
  return val;
}

int Marker::getMarkerId(Mat& markerImage,int &nRotations)
 {
  assert(markerImage.rows == markerImage.cols);//如果它的条件返回错误，则终止程序执行
  assert(markerImage.type() == CV_8UC1);

  Mat grey = markerImage;

  //Threshold image使用Otsu算法移除灰色的像素，只留下黑色和白色像素。
  //这是固定阀值方法  
  //输入图像image必须为一个2值单通道图像  
  //检测的轮廓数组，每一个轮廓用一个point类型的vector表示  
  //阀值  
  //max_value 使用 CV_THRESH_BINARY 和 CV_THRESH_BINARY_INV 的最大值  
  //type 
  threshold(grey,grey,125,255,THRESH_BINARY | THRESH_OTSU);//对候选标记区域的灰度图使用大律OSTU算法，求取二值化图，大范围图片用这个算法会影响性能。

#ifdef SHOW_DEBUG_IMAGES
  imshow("Binary marker",grey);
  imwrite("Binary marker.png",grey);   
#endif

  //所使用的标记都有一个内部的5x5编码，采用的是简单修改的汉明码。简单的说，就是5bits中只有2bits被使用，其他三位都是错误的识别码，也就是说我们至多有1024种不同的标识。我们的汉明码最大的不同是，汉明码的第一位（奇偶校验位的3和5)是反向的。所有ID 0(在汉明码是00000)，在这里是10000，目的是减少环境造成的影响.
  //标识被划分为7x7的网格，内部的5x5表示标识内容，额外的是黑色边界，接下来是逐个检查四条边的像素是否都是黑色的，若有不是黑色，那么就不是标识。
  int cellSize = markerImage.rows/7;

  for(int y=0;y<7;y++)
      {
      int inc = 6;

      if(y == 0 || y == 6) inc=1;//对第一行和最后一行，检查整个边界

      for(int x=0;x<7;x+=inc)
          {
          int cellX = x*cellSize;
          int cellY = y*cellSize;
          Mat cell = grey(Rect(cellX,cellY,cellSize,cellSize));

          int nZ = countNonZero(cell);//统计区域内非0的个数。

          if(nZ > (cellSize*cellSize)/2)
              {
              return -1;//如果边界信息不是黑色的，就不是一个标识。
            }
        }
    }

  Mat bitMatrix = Mat::zeros(5,5,CV_8UC1);

  //得到信息（对于内部的网格，决定是否是黑色或白色的）就是判断内部5x5的网格都是什么颜色的，得到一个包含信息的矩阵bitMatrix。
  for(int y=0;y<5;y++)
      {
      for(int x=0;x<5;x++)
          {
          int cellX = (x+1)*cellSize;
          int cellY = (y+1)*cellSize;
          Mat cell = grey(Rect(cellX,cellY,cellSize,cellSize));

          int nZ = countNonZero(cell);
          if(nZ > (cellSize*cellSize)/2)
            bitMatrix.at<uchar>(y,x) = 1;
        }
    }

  //检查所有的旋转
  Mat rotations[4];
  int distances[4];

  rotations[0] = bitMatrix;
  distances[0] = hammDistMarker(rotations[0]);//求没有旋转的矩阵的海明距离。

  pair<int,int> minDist(distances[0],0);//把求得的海明距离和旋转角度作为最小初始值对，每个pair都有两个属性值first和second

  for(int i=1;i<4;i++)//就是判断这个矩阵与参考矩阵旋转多少度。
      {
      //计算最近的可能元素的海明距离
      rotations[i] = rotate(rotations[i-1]);//每次旋转90度
      distances[i] = hammDistMarker(rotations[i]);

      if(distances[i] < minDist.first)
          {
          minDist.first = distances[i];
          minDist.second = i;//这个pair的第二个值是代表旋转几次，每次90度。
        }
    }

  nRotations = minDist.second;//这个是将返回的旋转角度值
  if(minDist.first == 0)//若海明距离为0,则根据这个旋转后的矩阵计算标识ID
      {
      return mat2id(rotations[minDist.second]);
    }

  return -1;
}

void Marker::drawContour(Mat& image,Scalar color) const//在图像上画线，描绘出轮廓。
  {
  float thickness = 2;

  line(image,points[0],points[1],color,thickness,CV_AA);
  line(image,points[1],points[2],color,thickness,CV_AA);
  line(image,points[2],points[3],color,thickness,CV_AA);//thickness线宽
  line(image,points[3],points[0],color,thickness,CV_AA);//CV_AA是抗锯齿
}
