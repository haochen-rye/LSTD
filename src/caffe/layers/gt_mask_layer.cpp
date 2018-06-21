#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <math.h>       /* tan,ceil,floor,sqrt */

#include "caffe/layers/gt_mask_layer.hpp"

namespace caffe {

template <typename Dtype>
void GTMaskLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const GTMaskParameter& gt_mask_param = this->layer_param_.gt_mask_param();
  height_ = gt_mask_param.height();
  width_ = gt_mask_param.width();
  feature_size_ = height_ * width_;
  batch_size_ = gt_mask_param.batch_size();
}

template <typename Dtype>
void GTMaskLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(batch_size_, 1, height_, width_);
}

template <typename Dtype>
void GTMaskLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* gt_data = bottom[0]->cpu_data();
  Dtype* mask_data = top[0]->mutable_cpu_data();
  caffe_set<Dtype>(top[0]->count(), 0, mask_data);
  CHECK_EQ(bottom[0]->shape(0), 1);
  CHECK_EQ(bottom[0]->shape(1), 1);
  CHECK_EQ(bottom[0]->shape(3), 9);
  const int num_instance = bottom[0]->shape(2);
  for (int i=0; i < num_instance; ++i){
    int start_idx = i * 9;
    int item_id = gt_data[start_idx];
    CHECK_LT(item_id, batch_size_);
    // int group_label = gt_data[start_idx + 1];
    float xmin = gt_data[start_idx + 3];
    float ymin = gt_data[start_idx + 4];
    float xmax = gt_data[start_idx + 5];
    float ymax = gt_data[start_idx + 6];
    float center_x = (xmin + xmax) * width_ /2.;
    float center_y = (ymin + ymax) * height_ /2.;
    float box_width = (xmax - xmin) * width_;
    float box_height = (ymax - ymin) * height_;
    float rotation = tan(gt_data[start_idx + 8]);
    // FILL the mask of bounding box with 1, others 0
    for (int j=0; j < height_; ++j){
      for (int k=0; k < width_; ++k){
        if (rotation) {
          if (i>floor(height_*xmin) && i<ceil(height_*xmax) &&
            j>floor(width_*ymin) && j<ceil(width_*ymax)){
              mask_data[item_id*feature_size_ + j*width_ + k] = 1;
          }            
        } else {
          float h_distance = (j-center_y - rotation*(i - center_x))/sqrt(1 + rotation*rotation);
          float w_distance = copysign(1.0,rotation)*(rotation*(j-center_y) + 
            (i - center_x))/sqrt(1 + rotation*rotation);
          if (h_distance <= ceil(box_height/2.) && 
              w_distance <= ceil(box_width/2.)){
            mask_data[item_id*feature_size_ + j*width_ + k] = 1;
          }            
        }
      }
    }
  }
}

INSTANTIATE_CLASS(GTMaskLayer);
REGISTER_LAYER_CLASS(GTMask);

}  // namespace caffe
