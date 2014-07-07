// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "pthread.h"
#include "hdf5.h"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param);
  virtual ~HDF5OutputLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  inline std::string file_name() const { return file_name_; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void SaveBlobs();

  std::string file_name_;
  hid_t file_id_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};


template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5DataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void LoadHDF5FileData(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

// TODO: DataLayer, ImageDataLayer, and WindowDataLayer all have the
// same basic structure and a lot of duplicated code.

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* DataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class DataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* DataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~DataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  bool output_labels_;
  Caffe::Phase phase_;
};

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* ImageDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class ImageDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* ImageDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void ShuffleImages();

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  Caffe::Phase phase_;
};


// This function is used to create a pthread that prefetches the window data.
template <typename Dtype>
void* WindowDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class WindowDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* WindowDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~WindowDataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
};

template <typename Dtype>
void* MaskedWindowDataLayerPrefetch(void* layer_pointer);


//A structure storing all window information from image
struct ImageWindows{
	std::string image_path;
	std::string mask_path;
	std::string sp_path;
	int image_size[3];
	vector<int> fg_windows_index;
	vector<int> bg_windows_index;

};

template <typename Dtype>
class MaskedWindowDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* MaskedWindowDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit MaskedWindowDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual ~MaskedWindowDataLayer();

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom){return;}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom){return;}
  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;

  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  vector<ImageWindows> image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, INDEX_IN_IMAGE, NUM };
  vector<vector<float> > windows_;
  
};

template <typename Dtype>
void* PiWindowDataLayerPrefetch(void* layer_pointer);


template <typename Dtype>
class PiWindowDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* PiWindowDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit PiWindowDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual ~PiWindowDataLayer();

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom){return;}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom){return;}
  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;

  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_box_;
  shared_ptr<Blob<Dtype> > prefetch_reg_;

  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  vector<ImageWindows> image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, INDEX_IN_IMAGE, NUM };
  vector<vector<float> > windows_;
  
};






}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
