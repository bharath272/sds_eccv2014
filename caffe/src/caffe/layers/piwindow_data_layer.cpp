// Copyright 2013 Ross Girshick

#include <stdint.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/sprep.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::string;

// caffe.proto > LayerParameter
//   'source' field specifies the window_file
//   'cropsize' indicates the desired warped size

// TODO(rbg):
//  - try uniform sampling over classes

namespace caffe {

template <typename Dtype>
void* PiWindowDataLayerPrefetch(void* layer_pointer) {
  PiWindowDataLayer<Dtype>* layer = 
      reinterpret_cast<PiWindowDataLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_box = layer->prefetch_box_->mutable_cpu_data();
  Dtype* top_reg = layer->prefetch_reg_->mutable_cpu_data();

  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.window_data_param().scale();
  const int batchsize = layer->layer_param_.window_data_param().batch_size();
  const int cropsize = layer->layer_param_.window_data_param().crop_size();
  const int context_pad = layer->layer_param_.window_data_param().context_pad();
  const bool mirror = layer->layer_param_.window_data_param().mirror();
  const float fg_fraction = layer->layer_param_.window_data_param().fg_fraction();
  const Dtype* mean = layer->data_mean_.cpu_data();
  const int mean_off = (layer->data_mean_.width() - cropsize) / 2;
  const int mean_width = layer->data_mean_.width();
  const int mean_height = layer->data_mean_.height();
  cv::Size cv_crop_size(cropsize, cropsize);
  const string& crop_mode = layer->layer_param_.window_data_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  memset(top_box, 0, sizeof(Dtype)*layer->prefetch_box_->count());
  memset(top_reg, 0, sizeof(Dtype)*layer->prefetch_reg_->count());
 

//  CHECK_EQ(mean_width, mean_height);
//  CHECK_EQ(mean_width, 256);
//  CHECK_EQ(mean_off, 14);

  //we will sample some number of images
  const int num_images=4;
  int num_samples[2];
  int itemid=0;
  //number per image
  const int num_per_img=static_cast<int>(static_cast<float>(batchsize)/(static_cast<float>(num_images)));
  for (int img_num =0; img_num<num_images; ++img_num) {
    
    //this is how many we want per image
    int num_to_take=num_per_img;
    if(img_num==num_images-1) {
	num_to_take = batchsize - num_per_img*(num_images-1);
    };
    if(num_to_take<0){
      break;
    }
     
    int num_fg = static_cast<int>(static_cast<float>(num_to_take) * fg_fraction);  
    num_samples[0]=num_to_take - num_fg;
    num_samples[1]=num_fg;

    //sample image
    ImageWindows imgwin;
    int numwin;
    int img_index;
    do{
      img_index=rand() % layer->image_database_.size();
      imgwin=layer->image_database_[img_index];
      numwin=imgwin.bg_windows_index.size()+imgwin.fg_windows_index.size();
      bool enough=(numwin>10);
      if(enough) {break;}
	}while((imgwin.bg_windows_index.size()+imgwin.fg_windows_index.size())<10);
    //read image
    cv::Mat cv_img = cv::imread(layer->image_database_[img_index].image_path, CV_LOAD_IMAGE_COLOR);
    if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << layer->image_database_[img_index].image_path;
        return (void*)NULL;
    }
    const int channels = cv_img.channels();

    //load superpixel image
    std::pair<cv::Mat, cv::Mat> sprep = load_sp_rep(imgwin.sp_path, imgwin.mask_path);
    cv::Mat sp = sprep.first;
    cv::Mat reg2sp =sprep.second;
    //sample windows
    for (int is_fg = 0; is_fg < 2; ++is_fg) {
      for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
        // sample a window
        int window_index= (is_fg)? imgwin.fg_windows_index[rand() % imgwin.fg_windows_index.size()]
                                 : imgwin.bg_windows_index[rand() % imgwin.bg_windows_index.size()];
        vector<float> window = layer->windows_[window_index];

 
        bool do_mirror = false;
        if (mirror && rand() % 2) {
          do_mirror = true;
        }
  
        // crop window out of image and warp it
        int x1 = window[PiWindowDataLayer<Dtype>::X1];
        int y1 = window[PiWindowDataLayer<Dtype>::Y1];
        int x2 = window[PiWindowDataLayer<Dtype>::X2];
        int y2 = window[PiWindowDataLayer<Dtype>::Y2];
	    int index_in_image = window[PiWindowDataLayer<Dtype>::INDEX_IN_IMAGE];  
        int pad_w = 0;
        int pad_h = 0;
        if (context_pad > 0 || use_square) {
          // scale factor by which to expand the original region 
          // such that after warping the expanded region to cropsize x cropsize
          // there's exactly context_pad amount of padding on each side
          Dtype context_scale = static_cast<Dtype>(cropsize) /
              static_cast<Dtype>(cropsize - 2*context_pad);
  
          // compute the expanded region
          Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
          Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
          Dtype center_x = static_cast<Dtype>(x1) + half_width;
          Dtype center_y = static_cast<Dtype>(y1) + half_height;
          if (use_square) {
            if (half_height > half_width) {
              half_width = half_height;
            } else {
              half_height = half_width;
            }
          }
          x1 = static_cast<int>(round(center_x - half_width*context_scale));
          x2 = static_cast<int>(round(center_x + half_width*context_scale));
          y1 = static_cast<int>(round(center_y - half_height*context_scale));
          y2 = static_cast<int>(round(center_y + half_height*context_scale));
          
          // the expanded region may go outside of the image
          // so we compute the clipped (expanded) region and keep track of
          // the extent beyond the image
          int unclipped_height = y2-y1+1;
          int unclipped_width = x2-x1+1;
          int pad_x1 = std::max(0, -x1);
          int pad_y1 = std::max(0, -y1);
          int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
          int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
          // clip bounds
          x1 = x1 + pad_x1;
          x2 = x2 - pad_x2;
          y1 = y1 + pad_y1;
          y2 = y2 - pad_y2;
          CHECK_GT(x1, -1);
          CHECK_GT(y1, -1);
          CHECK_LT(x2, cv_img.cols);
          CHECK_LT(y2, cv_img.rows);
  
          int clipped_height = y2-y1+1;
          int clipped_width = x2-x1+1;
  
          // scale factors that would be used to warp the unclipped 
          // expanded region
          Dtype scale_x = 
              static_cast<Dtype>(cropsize)/static_cast<Dtype>(unclipped_width);
          Dtype scale_y = 
              static_cast<Dtype>(cropsize)/static_cast<Dtype>(unclipped_height);
  
          // size to warp the clipped expanded region to
          cv_crop_size.width = 
              static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
          cv_crop_size.height = 
              static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
          pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
          pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
          pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
          pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));
  
          pad_h = pad_y1;
          // if we're mirroring, we mirror the padding too (to be pedantic)
          if (do_mirror) {
            pad_w = pad_x2;
          } else {
            pad_w = pad_x1;
          }
  
          // ensure that the warped, clipped region plus the padding
          // fits in the cropsize x cropsize image (it might not due to rounding)
          if (pad_h + cv_crop_size.height > cropsize) {
            cv_crop_size.height = cropsize - pad_h;
          }
          if (pad_w + cv_crop_size.width > cropsize) {
            cv_crop_size.width = cropsize - pad_w;
          }
        }
  
  //      CHECK_GT(x1, -1);
  //      CHECK_GT(y1, -1);
  //      CHECK_LT(x1, cv_img.cols);
  //      CHECK_LT(y1, cv_img.rows);
  //      CHECK_GT(x2, x1-1);
  //      CHECK_GT(y2, y1-1);
  //      CHECK_LT(x2, cv_img.cols);
  //      CHECK_LT(y2, cv_img.rows);
        cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
        cv::Mat cv_cropped_img = cv_img(roi);
        cv::resize(cv_cropped_img, cv_cropped_img, 
            cv_crop_size, 0, 0, cv::INTER_LINEAR);
	    cv::Size sz=sp.size();
	    cv::Mat cv_cropped_sp=sp(roi);
	    cv::resize(cv_cropped_sp, cv_cropped_sp, cv_crop_size, 0, 0, cv::INTER_NEAREST);
        // horizontal flip at random
        if (do_mirror) {
          cv::flip(cv_cropped_img, cv_cropped_img, 1);
	      cv::flip(cv_cropped_sp, cv_cropped_sp, 1);
        }
  
        // copy the warped window into top_data
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < cv_cropped_img.rows; ++h) {
            for (int w = 0; w < cv_cropped_img.cols; ++w) {
              Dtype pixel = 
                  static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);
              int spid=cv_cropped_sp.at<int>(h,w);
              bool belongs=(reg2sp.at<uchar>(spid-1,index_in_image-1) > 0);
              Dtype value=(pixel
                      - mean[(c * mean_height + h + mean_off + pad_h)
                             * mean_width + w + mean_off + pad_w])
                    * scale;
              top_box[((itemid * channels + c) * cropsize + h + pad_h) * cropsize + w + pad_w]
                  = value; 
              if(belongs)
              { 
                top_reg[((itemid * channels + c) * cropsize + h + pad_h) * cropsize + w + pad_w]
                  = value;
              }
            }
          }
        }
  
        // get window label
        top_label[itemid] = window[PiWindowDataLayer<Dtype>::LABEL];
  
          itemid++;
      }
    }

   



  }

  return (void*)NULL;
}

template <typename Dtype>
PiWindowDataLayer<Dtype>::~PiWindowDataLayer<Dtype>() {
  JoinPrefetchThread();
}


template <typename Dtype>
void PiWindowDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // SetUp runs through the window_file and creates two structures 
  // that hold windows: one for foreground (object) windows and one 
  // for background (non-object) windows. We use an overlap threshold 
  // to decide which is which.

  CHECK_EQ(bottom.size(), 0) << "Window data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 3) << "Window data Layer prodcues three blobs as output.";

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: " 
      << this->layer_param_.window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: " 
      << this->layer_param_.window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.window_data_param().fg_fraction();

  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file " 
      << this->layer_param_.window_data_param().source() << std::endl;

  vector<float> label_hist(21);
  std::fill(label_hist.begin(), label_hist.end(), 0);

  string hashtag;
  int image_index, channels;
  while (infile >> hashtag >> image_index) {
    CHECK_EQ(hashtag, "#");
    // read image path
    ImageWindows imgwin;
    
    string image_path;
    infile >> imgwin.image_path;
    //imgwin.image_path=image_path;

    string mask_path;
    infile >> imgwin.mask_path;
    string sp_path;
    infile >> imgwin.sp_path;
    // read image dimensions
    infile >> imgwin.image_size[0] >> imgwin.image_size[1] >> imgwin.image_size[2];
    channels = imgwin.image_size[0];
    
    //image_database_.push_back(std::make_pair(image_path, image_size));
    	
    // read each box
    int num_windows;
    infile >> num_windows;
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2, index_in_image;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2 >> index_in_image;

      vector<float> window(PiWindowDataLayer::NUM);
      window[PiWindowDataLayer::IMAGE_INDEX] = image_index;
      window[PiWindowDataLayer::LABEL] = label;
      window[PiWindowDataLayer::OVERLAP] = overlap;
      window[PiWindowDataLayer::X1] = x1;
      window[PiWindowDataLayer::Y1] = y1;
      window[PiWindowDataLayer::X2] = x2;
      window[PiWindowDataLayer::Y2] = y2;
      window[PiWindowDataLayer::INDEX_IN_IMAGE] = index_in_image;
      
      // add window to foreground list or background list
      if (overlap >= this->layer_param_.window_data_param().fg_threshold()) {
        CHECK_GT(window[PiWindowDataLayer::LABEL], 0);
        imgwin.fg_windows_index.push_back(windows_.size());
      } else if (overlap < this->layer_param_.window_data_param().bg_threshold()) {
        // background window, force label and overlap to 0
        window[PiWindowDataLayer::LABEL] = 0;
        window[PiWindowDataLayer::OVERLAP] = 0;
        imgwin.bg_windows_index.push_back(windows_.size());
      }
      windows_.push_back(window);
      label_hist[window[PiWindowDataLayer::LABEL]]++;
    }
    image_database_.push_back(imgwin);
    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_database_[image_index].image_path << " " 
          << imgwin.image_size[0] << " "
          << imgwin.image_size[1] << " "
          << imgwin.image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  }

  LOG(INFO) << "Number of images: " << image_index+1;

  for (int i = 0; i < 21; ++i) {
    LOG(INFO) << "class " << i << " has " << label_hist[i] << " samples";
  }

  LOG(INFO) << "Amount of context padding: " 
      << this->layer_param_.window_data_param().context_pad();

  LOG(INFO) << "Crop mode: " << this->layer_param_.window_data_param().crop_mode();

  // image
  int cropsize = this->layer_param_.window_data_param().crop_size();
  CHECK_GT(cropsize, 0);
  (*top)[0]->Reshape(
      this->layer_param_.window_data_param().batch_size(), channels, cropsize, cropsize);
  prefetch_box_.reset(new Blob<Dtype>(
      this->layer_param_.window_data_param().batch_size(), channels, cropsize, cropsize));
  (*top)[1]->Reshape(
      this->layer_param_.window_data_param().batch_size(), channels, cropsize, cropsize);
  prefetch_reg_.reset(new Blob<Dtype>(
      this->layer_param_.window_data_param().batch_size(), channels, cropsize, cropsize));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[2]->Reshape(this->layer_param_.window_data_param().batch_size(), 1, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.window_data_param().batch_size(), 1, 1, 1));

  // check if we want to have mean
  if (this->layer_param_.window_data_param().has_mean_file()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.window_data_param().mean_file();
    ReadProtoFromBinaryFile(this->layer_param_.window_data_param().mean_file().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.width(), data_mean_.height());
    CHECK_EQ(data_mean_.channels(), channels);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, channels, cropsize, cropsize);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_box_->mutable_cpu_data();
  prefetch_reg_->mutable_cpu_data();

  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, PiWindowDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void PiWindowDataLayer<Dtype>::CreatePrefetchThread() {
  const bool prefetch_needs_rand =
      this->layer_param_.window_data_param().mirror() ||
      this->layer_param_.window_data_param().crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, PiWindowDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void PiWindowDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int PiWindowDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}


template <typename Dtype>
Dtype PiWindowDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_box_->cpu_data(),
      sizeof(Dtype) * prefetch_box_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_reg_->cpu_data(),
      sizeof(Dtype) * prefetch_reg_->count());
  memcpy((*top)[2]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());

  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(PiWindowDataLayer);

}  // namespace caffe
