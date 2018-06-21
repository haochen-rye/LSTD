#include <functional>
#include <utility>
#include <algorithm>
#include <vector>

#include "caffe/layers/multiclass_sigmoid_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void MulticlassSigmoidLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  MulticlassSigmoidLossParameter multiclass_sigmoid_loss_param = 
    this->layer_param_.multiclass_sigmoid_loss_param();
  include_background_ = multiclass_sigmoid_loss_param.include_background();
  }

template <typename Dtype>
void MulticlassSigmoidLossLayer<Dtype>::Reshape(
    const std::vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  batch_size_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  CHECK_EQ(batch_size_, bottom[1]->num())
  	  << "Score and Label should habe the same batch_size";
  if (include_background_){
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1) + 1)
      << "Score and label should have the same num_classes";
    num_classes_ = channels_ - 1;
    score_start_idx_ = 1;
  } else {
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
      << "Score and label should have the same num_classes";
    num_classes_ = channels_;
    score_start_idx_ = 0;
  }
}

template <typename Dtype>
void MulticlassSigmoidLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < batch_size_; i++) {
  	for (int c = 0; c < num_classes_; c++) {
      Dtype score = input_data[i * channels_ + c + score_start_idx_];
      const int target = static_cast<int>(label[i * num_classes_ + c]);
      loss -= score * (target -(score >= 0)) -
        log(1 + exp(score - 2 * score * (score >= 0)));
  	}
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[1]->count();
}

template <typename Dtype>
void MulticlassSigmoidLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
  	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int i = 0; i < batch_size_; ++i) {
      for (int c = 0; c < num_classes_; c++) {
        int score_idx = i * channels_ + c + score_start_idx_;
        const int target = static_cast<int>(label[i * num_classes_ + c]);
        bottom_diff[score_idx] = sigmoid_output_data[score_idx] - target;
      }
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / bottom[1]->count(), bottom_diff);
  }
}
#ifdef CPU_ONLY
STUB_GPU(MulticlassSigmoidLossLayer);
#endif
INSTANTIATE_CLASS(MulticlassSigmoidLossLayer);
REGISTER_LAYER_CLASS(MulticlassSigmoidLoss);
}