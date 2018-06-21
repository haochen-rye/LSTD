#include <algorithm>
#include <functional>
#include <utility>
#include <string>
#include <vector>

// #include "caffe/layer.hpp"
// #include "caffe/util/io.hpp"
// #include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
// #include "caffe/util/bbox_util.hpp"
#include "caffe/layers/rela_oicr_layer.hpp"

namespace caffe {

template <typename Dtype>
void RelaOICRLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  OICRParameter oicr_param = 
    this->layer_param_.oicr_param();

  num_proposal_ = oicr_param.num_proposal();
  num_classes_ = oicr_param.num_classes();
  num_rpn_ = oicr_param.num_rpn();
  merge_proposal_ = oicr_param.merge_proposal();
  background_label_id_ = oicr_param.background_label_id();
  overlap_threshold_ = oicr_param.overlap_threshold();
  include_background_ = oicr_param.include_background();
  output_bbox_ = oicr_param.output_bbox();
  include_ohem_ = oicr_param.include_ohem();
  min_overlap_ = oicr_param.min_overlap();
  pos_neg_ratio_ = oicr_param.pos_neg_ratio();
  del_neg_ = oicr_param.del_neg();
  use_gt_score_ = oicr_param.use_gt_score();
  use_max_score_threshold_ = oicr_param.use_max_score_threshold();
  max_score_threshold_ = oicr_param.max_score_threshold();
  // const int total_num_propoal = bottom[2]->shape(0);
  batch_size_ = bottom[0]->shape(0);
  channels_ = bottom[2]->shape(2);

  if (include_background_){
    actual_classes_ = channels_ - 1;
  CHECK_EQ(bottom[0]->shape(1) , actual_classes_)
    << "The classes of label and score should match";    
  } else {
    actual_classes_ = channels_;
  CHECK_EQ(bottom[0]->shape(1), actual_classes_)
    << "The input label blob should have the standard format";
  }

  top[0]->Reshape(batch_size_, num_proposal_, 1, 1);
  top[1]->Reshape(batch_size_, num_proposal_, 1, 1);
  if (output_bbox_){
// Fake reshape the label output
    top[2]->Reshape(1,8,1,1);
  }

}

template <typename Dtype>
void RelaOICRLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[2]->shape(0), bottom[3]->shape(0));
  CHECK_EQ(bottom[2]->shape(1), bottom[3]->shape(1));
  CHECK_EQ(bottom[2]->shape(2), bottom[3]->shape(2));
  CHECK_EQ(bottom[2]->shape(3), bottom[3]->shape(3));
  }

template <typename Dtype>
void RelaOICRLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // batch_size is indeed the num_Img \times the num_rpn
  const int actual_batch_size = batch_size_ / num_rpn_;
  const int total_num_propoal = batch_size_ * num_proposal_ ;
  CHECK_EQ(bottom[2]->shape(1), num_proposal_)
    << "The total_num_propoal mismatch for input score blob";  
  CHECK_EQ(bottom[1]->shape(2), total_num_propoal)
    << "The total_num_propoal mismatch for input proposal blob";
  CHECK_EQ(bottom[2]->shape(0), batch_size_)
    << "The total_num_propoal mismatch for input score blob";
  CHECK_EQ(bottom[1]->shape(3), 7)
    << "The input proposal blob should have the standard format";
  CHECK_EQ(bottom[2]->shape(2), num_classes_)
    << "The input score blob should have ";

  const Dtype* bottom_label = bottom[0]->cpu_data();
  const Dtype* bottom_proposal = bottom[1]->cpu_data();
  const Dtype* first_score = bottom[2]->cpu_data();
  const Dtype* second_score = bottom[3]->cpu_data();
  const int score_num = bottom[3]->count(1);
  Blob<Dtype> rela_score;
  rela_score.CopyFrom(*bottom[2], false, true);
  Dtype* rela_score_data = rela_score.mutable_cpu_data();
  caffe_sub(bottom[2]->count(), second_score, first_score, rela_score_data);

  Dtype* top_label = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_label);
  Dtype* top_label_weight = top[1]->mutable_cpu_data();
  caffe_set(top[1]->count(), Dtype(0), top_label_weight);

  LabelBBox all_gt_bboxes;
  int num_instance = 0;
  GetGroundTruth(bottom_label, bottom_proposal, 
    second_score, rela_score_data,
    batch_size_,channels_, actual_classes_, num_proposal_,
    max_score_threshold_,
    background_label_id_, include_background_, num_instance,
    &all_gt_bboxes);

  if (output_bbox_){
    top[2]->Reshape(num_instance, 7 ,1,1);
    Dtype* top_bbox = top[2]->mutable_cpu_data();
    int count = 0;
    for(int i = 0; i < batch_size_; ++i){
      const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes.find(i)->second;
      int num_gt = gt_bboxes.size();
      if (num_gt == 0){
        continue;
      } else {
        for (int j = 0; j < num_gt; ++j){
          int start_index = (count + j) * 7;
          top_bbox[start_index] = i;
          top_bbox[start_index + 1] = gt_bboxes[j].label();
          top_bbox[start_index + 2] = gt_bboxes[j].score();
          top_bbox[start_index + 3] = gt_bboxes[j].xmin();
          top_bbox[start_index + 4] = gt_bboxes[j].ymin();
          top_bbox[start_index + 5] = gt_bboxes[j].xmax();
          top_bbox[start_index + 6] = gt_bboxes[j].ymax();
          }
        count += num_gt;
      }
    }
  }

  LabelBBox all_detections;
  GetDetectionResults(bottom_proposal, bottom[1]->shape(2), 
    num_proposal_,  actual_batch_size,
    background_label_id_, &all_detections);

// Match the predictions with the ground truth and the align the label.
  for (int i = 0; i < batch_size_ ; ++i){
    const vector<NormalizedBBox>& img_detections = all_detections.find(i)->second;
    const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes.find(i)->second;

  vector<float> match_overlaps;
  MatchBBOX(gt_bboxes, img_detections, second_score + i * score_num,
      use_gt_score_, channels_, include_background_,
      overlap_threshold_, min_overlap_,
      include_ohem_, pos_neg_ratio_, del_neg_,
      top_label + i * num_proposal_, 
      top_label_weight + i * num_proposal_,
      &match_overlaps );   
  }

  // caffe_copy(batch_size_, &align_label, top_label);
}

INSTANTIATE_CLASS(RelaOICRLayer);
REGISTER_LAYER_CLASS(RelaOICR);

}  // namespace caffe
