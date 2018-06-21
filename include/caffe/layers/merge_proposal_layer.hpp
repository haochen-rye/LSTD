#ifndef CAFFE_MERGE_PROPOSAL_LAYER_HPP_
#define CAFFE_MERGE_PROPOSAL_LAYER_HPP_


#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

/**
 * @brief Generate the detection output based on location and confidence
 * predictions by doing non maximum suppression.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class MergeProposalLayer : public Layer<Dtype> {
 public:
  explicit MergeProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MergeProposal"; }
  virtual inline int ExactBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times N \times 7) @f$
   *      N is the number of detections after nms, and each row is:
   *      [image_id, label, confidence, xmin, ymin, xmax, ymax]
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }
  //   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //   return;
  // }

  int num_rpn_;
  int batch_size_;
  int num_proposal_;

};

}  // namespace caffe

#endif  // CAFFE_MERGE_PROPOSAL_LAYER_HPP_
