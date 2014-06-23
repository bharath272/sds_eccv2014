#include <stdint.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <fstream>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/sprep.h"
using std::string;

namespace caffe {

//load superpixels
std::pair<cv::Mat, cv::Mat> load_sp_rep(const string& sp_path, const string& mask_path) {
  std::ifstream fin(sp_path.c_str());
  unsigned int spsize[2];
  fin >> spsize[0] >> spsize[1];
  cv::Mat sp=cv::Mat(spsize[0], spsize[1], CV_32SC1);
  unsigned int tmp;
  for(int j=0; j<spsize[1];j++)
  {
    for(int i=0; i<spsize[0]; i++)
    {
      fin>>tmp;
      sp.at<int>(i,j)=tmp;	
    }
  } 	

  fin.close();
  cv::Size sz=sp.size();
  cv::Mat reg2sp = cv::imread(mask_path, CV_LOAD_IMAGE_GRAYSCALE);
  return std::make_pair(sp, reg2sp);

}

}
