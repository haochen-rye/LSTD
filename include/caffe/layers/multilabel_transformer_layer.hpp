#ifndef CAFFE_MULTILABEL_TRANSFORMER_LAYER_HPP_
#define CAFFE_MULTILABEL_TRANSFORMER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

// MULTILABEL_TRANSFORMER LAYER: 
  // INPUT: LABEL(1 * 1 * N * 7 ,img_id, label, object_id, xmin, ymin, xmax, ymax)
  // OUTPUT: MULTILABEL(n * 20 )
template <typename Dtype>
class MultilabelTransformerLayer : public Layer<Dtype> {
 public:
  explicit MultilabelTransformerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "MultilabelTransformer"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- MultilabelAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  }

  int batch_size_;
  int num_classes_;
  int num_rpn_;

};

}  // namespace caffe

#endif  // CAFFE_MultilabelTRANSFORMER_LAYER_HPP_

