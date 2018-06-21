#ifndef CAFFE_MULTICLASSACCURACY_LAYER_HPP_
#define CAFFE_MULTICLASSACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MulticlassAccuracyLayer : public Layer<Dtype> {
 public:
  explicit MulticlassAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "MulticlassAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- MulticlassAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  bool include_background_;
  int score_start_idx_;
  int batch_size_;
  // channels_ is the dim for score, num_classes_ is the dim for label
  int channels_;
  int num_classes_;

  };

}  // namespace caffe

#endif  // CAFFE_MULTICLASSACCURACY_LAYER_HPP_
