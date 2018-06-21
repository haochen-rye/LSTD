#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/weighted_softmax_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedSoftmaxWithCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  } 
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  // softmax_bottom_vec_.clear();
  // softmax_bottom_vec_.push_back(bottom[0]);
  // softmax_top_vec_.clear();
  // softmax_top_vec_.push_back(&prob_);
  // softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);  

  softmax_predict_bottom_vec_.clear();
  softmax_predict_bottom_vec_.push_back(bottom[0]);
  softmax_target_bottom_vec_.clear();
  softmax_target_bottom_vec_.push_back(bottom[1]);
  softmax_predict_top_vec_.clear();
  softmax_predict_top_vec_.push_back(&predict_prob_);
  softmax_target_top_vec_.clear();
  softmax_target_top_vec_.push_back(&target_prob_);  

  softmax_layer_->SetUp(softmax_predict_bottom_vec_, softmax_predict_top_vec_);

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void WeightedSoftmaxWithCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  // softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_layer_->Reshape(softmax_predict_bottom_vec_, softmax_predict_top_vec_); 
  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes())
      << "bottom[0] and bottom[1] should have the same shape.";
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
      CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i))
          << "bottom[0] and bottom[1] should have the same shape.";
  }

  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  // CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
  //     << "Number of labels must match number of predictions; "
  //     << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
  //     << "label count (number of labels) must be N*H*W, "
  //     << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void WeightedSoftmaxWithCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_predict_bottom_vec_, softmax_predict_top_vec_);
  softmax_layer_->Forward(softmax_target_bottom_vec_, softmax_target_top_vec_);  
  const Dtype* predict_prob_data = predict_prob_.cpu_data();
  const Dtype* target_prob_data = target_prob_.cpu_data();

  int dim = predict_prob_.count() / outer_num_;
  Dtype loss = 0;

  int count = 0; 
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      ++count;
      for (int k = 0; k < predict_prob_.shape(softmax_axis_); ++k) {
        loss -= target_prob_data[i * dim + k * inner_num_ + j]
            * log(std::max(predict_prob_data[i * dim + k * inner_num_ + j], Dtype(FLT_MIN)));
      }
    }
  }
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      for (int k = 0; k < predict_prob_.shape(softmax_axis_); ++k) {
        loss += target_prob_data[i * dim + k * inner_num_ + j]
            * log(std::max(target_prob_data[i * dim + k * inner_num_ + j], Dtype(FLT_MIN)));
      }
    }
  }

  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
      normalization_, outer_num_, inner_num_, count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
  if (top.size() == 2) {
    top[1]->ShareData(predict_prob_);
  }
}

template <typename Dtype>
void WeightedSoftmaxWithCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* predict_prob_data = predict_prob_.cpu_data();
    const Dtype* target_prob_data = target_prob_.cpu_data();
    // int dim = predict_prob_.count() / outer_num_;
    caffe_sub(predict_prob_.count(), predict_prob_data, target_prob_data, bottom_diff);
    caffe_mul(predict_prob_.count(), target_prob_data, bottom_diff, bottom_diff);
    // Scale gradient
    // Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
    //     normalization_, outer_num_, inner_num_, count);
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
    normalization_, outer_num_, inner_num_, outer_num_ * inner_num_);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    // caffe_scal(prob_.count(), loss_weight, bottom_diff);
    caffe_scal(predict_prob_.count(),loss_weight,bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedSoftmaxWithCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(WeightedSoftmaxWithCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(WeightedSoftmaxWithCrossEntropyLoss);

}  // namespace caffe
