#ifndef CAFFE_MULTI_CLASS_SIGMOID_LOSS_LAYER_HPP_
#define CAFFE_MULTI_CLASS_SIGMOID_LOSS_LAYER_HPP_

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
class MulticlassSigmoidLossLayer : public LossLayer<Dtype> {

 public:
  explicit MulticlassSigmoidLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "MulticlassSigmoidLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<Layer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;

  bool include_background_;
  int score_start_idx_;
  int batch_size_;
  // channels_ is the dim for score, num_classes_ is the dim for label
  int channels_;
  int num_classes_;
  };
  
}  // namespace caffe
#endif  // CAFFE_MULTI_CLASS_SIGMOIDLOSS_LAYER_HPP_