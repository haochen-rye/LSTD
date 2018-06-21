#ifndef CAFFE_GT_MASK_LAYER_HPP_
#define CAFFE_GT_MASK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * Generate featue mask for ground truth bounding box
 */
template <typename Dtype>
class GTMaskLayer : public Layer<Dtype> {
 public:
  explicit GTMaskLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline int ExactNumBottomBlobs() const { return 1; }  
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline const char* type() const { return "GTMask"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times N \times 9) @f$
   *    The boundary of bounding box:  [item_id, group_label, instance_id,
    xmin, ymin, xmax, ymax, difficult, rotation]
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times 1 \times H \times W) @f$
   *      the computed outputs @f$
   *        N is the batch_size,
            H and W is feature map size
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  };

  int height_;
  int width_;
  int feature_size_;
  int batch_size_;
};
  
}  // namespace caffe

#endif  // CAFFE_GT_MASK_LAYER_HPP_
