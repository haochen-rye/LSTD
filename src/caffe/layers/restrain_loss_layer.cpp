// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/restrain_loss_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::abs;

namespace caffe {

template <typename Dtype>
void RestrainLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    }  
  RestrainLossParameter restrain_loss_param = this->layer_param_.restrain_loss_param();
  spatial_scale_ = restrain_loss_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void RestrainLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape); 
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();  
  feature_dim_ = height_ * width_;
  count_ = bottom[0]->count();
  num_rois_ = bottom[1]->shape(2);
}

template <typename Dtype>
void RestrainLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bg_blob_.CopyFrom(*bottom[0],false,true); 
  // Dtype* bg_data = bg_blob_.mutable_cpu_data();
  Dtype* bg_data = static_cast<Dtype*>(bg_blob_.mutable_cpu_data());
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int batch_size = bottom[0]->num();
  int fg_count = 0;

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois_; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = max((int)round(bottom_rois[3] * spatial_scale_),0);
    int roi_start_h = max((int)round(bottom_rois[4] * spatial_scale_),0);
    int roi_end_w = min((int)round(bottom_rois[5] * spatial_scale_),width_);
    int roi_end_h = min((int)round(bottom_rois[6] * spatial_scale_),height_);
    fg_count += (roi_end_h - roi_start_h) * (roi_end_w - roi_start_w) ;
    // LOG(INFO) << "fg_count: " << fg_count;

    // int roi_start_w = round(bottom_rois[3] * spatial_scale_);
    // int roi_start_h = round(bottom_rois[4] * spatial_scale_);
    // int roi_end_w = round(bottom_rois[5] * spatial_scale_);
    // int roi_end_h = round(bottom_rois[6] * spatial_scale_); 

    // LOG(INFO) << "roi_batch_ind: " << roi_batch_ind
    //   << " roi_start_w: " << roi_start_w
    //   << " roi_end_w: " << roi_end_w
    //   << " roi_start_h " << roi_start_h
    //   << " roi_end_h" << roi_end_h;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);
    CHECK_LE(roi_start_h, roi_end_h);
    CHECK_LE(roi_start_w, roi_end_w);

    // Dtype* batch_data = bg_data + bg_blob_.offset(roi_batch_ind);
    for (int c=0; c < channels_; ++c){
      for ( int h = roi_start_h; h < roi_end_h; ++h)
        for (int w = roi_start_w; w < roi_end_w; ++w){
          const int index =bg_blob_.offset(roi_batch_ind,c,h,w);
          // // CHECK_LE(index,bg_blob_.count());
          bg_data[index] = 0;
        }
    }

    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(0,0,1);
  }
  Dtype dot = caffe_cpu_dot(count_, bg_blob_.cpu_data(), bg_blob_.cpu_data() );
  bg_count_ = count_ - fg_count * channels_;
  // LOG(INFO) << "bg_count: " << bg_count_;
  Dtype loss = dot / bg_count_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss ;
}

template <typename Dtype>
void RestrainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] / bg_count_;
    caffe_cpu_axpby(
      count_, alpha, bg_blob_.cpu_data(), Dtype(0), bottom[0]->mutable_cpu_diff() );
  }

  if (propagate_down[1]) {
    LOG(FATAL) << this->type() <<" Layer cannot backpropagate to label input";
  }
}


#ifdef CPU_ONLY
STUB_GPU(RestrainLossLayer);
#endif

INSTANTIATE_CLASS(RestrainLossLayer);
REGISTER_LAYER_CLASS(RestrainLoss);

}  // namespace caffe
