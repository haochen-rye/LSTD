#ifndef CAFFE_GLOBAL_SUM_POOLING_LAYER_HPP_
#define CAFFE_GLOBAL_SUM_POOLING_LAYER_HPP_

#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class GlobalSumPoolingLayer : public Layer<Dtype> {
 public:
  explicit GlobalSumPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GlobalSumPooling"; }

  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_classes_;
  int actual_num_cls_;
  int cls_start_idx_;
  int num_proposal_;
  int inner_num_;
  bool include_background_;
  int batch_size_;
};

}  // namespace caffe

#endif  // CAFFE_GLOBAL_SUM_POOLING_LAYER_HPP_
