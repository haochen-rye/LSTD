#include <vector>

#include "caffe/layers/restrain_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template  <typename Dtype>
__global__ void SetZero(const int nthreads,
  Dtype* bottom_data,const int channels, const int height, const int width,
  const int roi_start_h, const int roi_end_h, 
  const int roi_start_w, const int roi_end_w){
  CUDA_KERNEL_LOOP(index, nthreads){
    int c = (index / width / height) / channels;
    bottom_data += c * height * width;
    for (int h = roi_start_h; h < roi_end_h; ++h){
      for (int w =roi_start_w; h < roi_end_w; ++w){
            const int index = h * width + w;
            bottom_data[index] = 0;
      }
    }
  }
}


template <typename Dtype>
void RestrainLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  bg_blob_.CopyFrom(*bottom[0],false,true); 
  Dtype* bg_data = static_cast<Dtype*>(bg_blob_.mutable_gpu_data());
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  // Number of ROIs
  int batch_size = bottom[0]->num();
  int feature_dim3 = bottom[0]->shape(1) * bottom[0]->shape(2) *bottom[0]->shape(3);
  CHECK_EQ( batch_size * feature_dim3, count_);
  int fg_count = 0;

// set the feature map in the object area to zero
  for (int n = 0; n < num_rois_; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = max((int)round(bottom_rois[3] * spatial_scale_),0);
    int roi_start_h = max((int)round(bottom_rois[4] * spatial_scale_),0);
    int roi_end_w = min((int)round(bottom_rois[5] * spatial_scale_),width_);
    int roi_end_h = min((int)round(bottom_rois[6] * spatial_scale_),height_);
    fg_count += (roi_end_h - roi_start_h) * (roi_end_w - roi_start_w);

    Dtype* batch_data = bg_data + bottom[0]->offset(roi_batch_ind);
    SetZero<Dtype><<<CAFFE_GET_BLOCKS(feature_dim3), CAFFE_CUDA_NUM_THREADS>>>(
        feature_dim3, batch_data, channels_, height_, width_,
        roi_start_h, roi_end_h, roi_start_w, roi_end_w);
    CUDA_POST_KERNEL_CHECK;
}
  Dtype dot;
  caffe_gpu_dot(count_, bg_blob_.gpu_data(), bg_blob_.gpu_data(), &dot );
  bg_count_ = count_ - fg_count * channels_;
  Dtype loss = dot /  Dtype(2 * bg_count_);
  top[0]->mutable_gpu_data()[0] = loss;
}

template <typename Dtype>
void RestrainLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype alpha = top[0]->gpu_diff()[0] / bg_count_;
    caffe_gpu_axpby(
      count_, alpha, bg_blob_.gpu_data(), Dtype(0), bottom[0]->mutable_gpu_diff() );
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() <<" Layer cannot backpropagate to label input";
  }  
}

INSTANTIATE_LAYER_GPU_FUNCS(RestrainLossLayer);

}  // namespace caffe
