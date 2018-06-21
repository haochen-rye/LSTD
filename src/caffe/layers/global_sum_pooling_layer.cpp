#include "caffe/layers/global_sum_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  GlobalSumPoolingParameter global_sum_pooling_param = 
    this->layer_param_.global_sum_pooling_param();
  num_classes_ = global_sum_pooling_param.num_classes();
  actual_num_cls_ = num_classes_;
  cls_start_idx_ = 0;
  num_proposal_ = global_sum_pooling_param.num_proposal();
  include_background_ = global_sum_pooling_param.include_background();
  inner_num_ = num_classes_ * num_proposal_;
  batch_size_ = bottom[0]->shape(0);
  if (include_background_){
    actual_num_cls_ = num_classes_ - 1;
    cls_start_idx_ = 1;
  } 
  top[0]->Reshape(batch_size_, actual_num_cls_, 1,1);
}

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set( top[0]->count(), Dtype(0), top_data);
  CHECK_EQ(num_proposal_, bottom[0]->shape(1))
    << "Channels of score blob must be num_proposal_";
  CHECK_EQ(num_classes_ , bottom[0]->shape(2))
    << "Height of score must be num_classes";

// Compute the class score for each image 
  for (int i = 0; i < batch_size_; ++i){
    for (int j = 0; j < num_proposal_; ++j){
      for (int k = cls_start_idx_; k < num_classes_; ++k){
        top_data[i * actual_num_cls_ + k - cls_start_idx_] += 
          bottom_data[i * inner_num_ + j * num_classes_ + k];          
      }
    }
  }

}

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Propagate to bottom
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  
  for (int i = 0; i < batch_size_; ++i){
    for (int j = 0; j < num_proposal_; ++j){
      for (int k = cls_start_idx_; k < num_classes_; ++k){
        bottom_diff[i * inner_num_ + j * num_classes_ + k] =
          top_diff[i * actual_num_cls_ + k - cls_start_idx_] ;
      }
    }
  }

  }
}


#ifdef CPU_ONLY
STUB_GPU(GlobalSumPoolingLayer);
#endif

INSTANTIATE_CLASS(GlobalSumPoolingLayer);
REGISTER_LAYER_CLASS(GlobalSumPooling);

}  // namespace caffe
