#ifndef CAFFE_OICR_LAYER_HPP_
#define CAFFE_OICR_LAYER_HPP_

#include <vector>
#include <string>
#include <map>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

// OICR LAYER: 
//   INPUT:
//        BOTTOM[0]:  LABEL ( n \times C )
//             *  n is the batch_size

//        BOTTOM[1]: PROPOSAL (1 \times 1 \times N \times 7) @f$
//            *      N is the number of detections after nms, and each row is:
//            *      [image_id, label, confidence, xmin, ymin, xmax, ymax]

//        BOTTOM[2]: SCORE (n \times num_proposal \times C)
//          * N is the number of proposals
//          * C is the number of classes

//   OUTPUT: LABEL FOR THE PROPOSAL(N \times C, 1, 1)
//           BBOX for the image (m \times 8 \times 1 \times 1)
//            * [image_id, label, confidence, xmin, ymin, xmax, ymax]


template <typename Dtype>
class OICRLayer : public Layer<Dtype> {
 public:
  explicit OICRLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "OICR"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  // virtual inline int ExactNumTopBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- NOT PROPOGATE BACK FOR LABEL.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  }

  int num_proposal_;
  int num_classes_;
  int batch_size_;
  int channels_;
  int actual_classes_;
  int num_rpn_;
  bool merge_proposal_;
  int background_label_id_;
  float overlap_threshold_;
  bool include_background_;
  bool output_bbox_;
  bool include_ohem_;
  float min_overlap_;
  float pos_neg_ratio_;
  bool del_neg_;
  bool use_gt_score_;
  bool use_max_score_threshold_;
  float max_score_threshold_;
};

}  // namespace caffe

#endif  // CAFFE_OICR_LAYER_HPP_

