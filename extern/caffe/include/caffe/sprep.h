#ifndef SPREP_H
#define SPREP_H 1
#include <stdint.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <fstream>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::string;
namespace caffe{
std::pair<cv::Mat, cv::Mat> load_sp_rep(const string& sp_path, const string& mask_path); 



}
#endif
