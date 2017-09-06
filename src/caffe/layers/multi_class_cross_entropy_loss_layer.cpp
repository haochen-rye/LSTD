#include "caffe/layers/multi_class_cross_entropy_loss_layer.hpp"
namespace caffe {
template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::Reshape(
    const std::vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  batch_size = bottom[0]->num();
  channels = bottom[0]->channels();
  CHECK_EQ(batch_size, bottom[1]->num())
  	  << "Number of labels must match number of predictions.";
}
template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < batch_size; i++) {
  	for (int c = 0; c < channels; c++) {
      Dtype tmp = input_data[i * channels + c];
      if (tmp < 1e-10) {
        tmp = 1e-10;
      }
      if (tmp > (1 - 1e-10)) {
        tmp = 1 - 1e-10;
      }
      loss -= label[i * channels + c] * log(tmp)
              + (1.0 - label[i * channels + c]) * log(1 - tmp);
  	}
  }
  top[0]->mutable_cpu_data()[0] = loss / batch_size;
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
    
    for (int i = 0; i < batch_size; ++i) {
      for (int c = 0; c < channels; c++) {
        Dtype tmp = input_data[i * channels + c];
        if (tmp < 1e-10) {
          tmp = 1e-10;
        }
        if (tmp > (1 - 1e-10)) {
          tmp = 1 - 1e-10;
        }
        bottom_diff[i * channels + c] = (tmp - label[i * channels + c])
                                        / (channels * tmp * (1 - tmp));
      }
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / batch_size, bottom_diff);
  }
}
#ifdef CPU_ONLY
STUB_GPU(MulticlassCrossEntropyLossLayer);
#endif
INSTANTIATE_CLASS(MulticlassCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MulticlassCrossEntropyLoss);
}