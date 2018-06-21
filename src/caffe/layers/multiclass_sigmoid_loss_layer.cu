#include "caffe/layers/multiclass_sigmoid_loss_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void MulticlassSigmoidLossForwardGPU(const int nthreads,
        const int channels, const int num_classes, const int score_start_idx,
          const Dtype* input_data, const Dtype* label, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int class_idx = index % num_classes;
    int batch_idx = index / num_classes;
    Dtype score = input_data[batch_idx * channels + class_idx + score_start_idx];
    loss[index] =  score * (label[index] -(score >= 0)) -
        log(1 + exp(score - 2 * score * (score >= 0)));
  }
}
template <typename Dtype>
void MulticlassSigmoidLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int nthreads = batch_size_ * num_classes_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_set(bottom[0]->count(), Dtype(0), loss_data);
  MulticlassSigmoidLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
        channels_, num_classes_, score_start_idx_, 
        input_data, label, loss_data);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  loss /= batch_size_ * num_classes_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void MulticlassSigmoidLossBackwardGPU(const int nthreads, 
    const int channels, const int num_classes, const int score_start_idx,
          const Dtype* input_data, const Dtype* label, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int class_idx = index % num_classes;
    int batch_idx = index / num_classes;
    int score_index = batch_idx * channels + class_idx + score_start_idx;
    Dtype score = input_data[score_index];
    bottom_diff[score_index] = score - label[index];
  }
}

template <typename Dtype>
void MulticlassSigmoidLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int nthreads = batch_size_ * num_classes_;

    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
    
    MulticlassSigmoidLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, channels_, num_classes_, 
          score_start_idx_, input_data, label, bottom_diff );
    const Dtype loss_weight = top[0]->cpu_diff()[0] / bottom[1]->count();
    
    caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(MulticlassSigmoidLossLayer);
}  // namespace caffe