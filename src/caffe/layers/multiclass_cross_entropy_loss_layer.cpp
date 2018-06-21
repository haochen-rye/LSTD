#include <functional>
#include <utility>

#include "caffe/layers/multiclass_cross_entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
    MulticlassCrossEntropyLossParameter multiclass_cross_entropy_loss_param = 
      this->layer_param_.multiclass_cross_entropy_loss_param();
    include_background_ = multiclass_cross_entropy_loss_param.include_background();
  }

template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::Reshape(
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
void MulticlassCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < batch_size_; i++) {
  	for (int c = 0; c < num_classes_; c++) {
      Dtype tmp = input_data[i * channels_ + c + score_start_idx_];
      if (tmp < 1e-10) {
        tmp = 1e-10;
      }
      if (tmp > (1 - 1e-10)) {
        tmp = 1 - 1e-10;
      }
      // loss -= label[i * num_classes_ + c] * log(tmp)
      //         + (1.0 - label[i * num_classes_ + c]) * log(1 - tmp);
      int label_index = i * num_classes_ + c;
      if(int(label[label_index]) == 0){
        loss -= log(1  - tmp);
      } else {
        loss -= log(tmp);
      }
  	}
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[1]->count();
}

template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
  	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int i = 0; i < batch_size_; ++i) {
      for (int c = 0; c < num_classes_; c++) {
        Dtype tmp = input_data[i * channels_ + c + score_start_idx_];
        if (tmp < 1e-10) {
          tmp = 1e-10;
        }
        if (tmp > (1 - 1e-10)) {
          tmp = 1 - 1e-10;
        }
        int label_index = i * num_classes_ + c;
        if (int(label[label_index]) == 0){
          bottom_diff[i * channels_ + c + score_start_idx_] = 1.0 / (1 - tmp);
        } else {
          bottom_diff[i * channels_ + c + score_start_idx_] = -1.0 / tmp;
        }
      }
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / bottom[1]->count(), bottom_diff);
  }
}
#ifdef CPU_ONLY
STUB_GPU(MulticlassCrossEntropyLossLayer);
#endif
INSTANTIATE_CLASS(MulticlassCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MulticlassCrossEntropyLoss);
}