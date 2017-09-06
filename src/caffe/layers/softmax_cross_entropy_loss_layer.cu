#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* predict_prob_data,const Dtype* target_prob_data,
          Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      loss[index] = -target_prob_data[index]
          * log(max(predict_prob_data[index], Dtype(FLT_MIN)));
  }
}


template <typename Dtype>
void SoftmaxWithCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  softmax_layer_->Forward(softmax_predict_bottom_vec_, softmax_predict_top_vec_);
  softmax_layer_->Forward(softmax_target_bottom_vec_, softmax_target_top_vec_);
  const Dtype* predict_prob_data = predict_prob_.gpu_data();
  const Dtype* target_prob_data = target_prob_.gpu_data();
  const int nthreads = predict_prob_.count();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, predict_prob_data,
                                target_prob_data, loss_data);
  Dtype loss_cross_entropy;
  caffe_gpu_asum(nthreads, loss_data, &loss_cross_entropy);

  SoftmaxCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, target_prob_data,
                                target_prob_data, loss_data);

  Dtype loss_entropy;
  caffe_gpu_asum(nthreads, loss_data, &loss_entropy);

  Dtype loss;
  loss = loss_cross_entropy - loss_entropy;
  
  // if (normalize_) {
  //   loss /= (outer_num_ * inner_num_);
  // } else {
  //   loss /= outer_num_;
  // }
  // top[0]->mutable_cpu_data()[0] = loss;

  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
      normalization_, outer_num_, inner_num_, valid_count);

  top[0]->mutable_cpu_data()[0] = loss / normalizer;  

  if (top.size() == 2) {
    top[1]->ShareData(predict_prob_);
  }
}


template <typename Dtype>
__global__ void SoftmaxCrossEntropyLossBackwardGPU(const int nthreads,
          const Dtype* predict_prob_data, const Dtype* target_prob_data,
          Dtype* bottom_diff, const int dim, const int spatial_dim) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    Dtype sum = 0;
    for (int k = 0; k < channels; ++k) {
      sum += target_prob_data[n * dim + k * spatial_dim + s];
    }
    for (int k = 0; k < channels; ++k) {
        bottom_diff[n * dim + k * spatial_dim + s] =
            sum * predict_prob_data[n * dim + k * spatial_dim + s]
            - target_prob_data[n * dim + k * spatial_dim + s];
    }
  }
}

template <typename Dtype>
void SoftmaxWithCrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* predict_prob_data = predict_prob_.gpu_data();
    const Dtype* target_prob_data = target_prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_sub(predict_prob_.count(), predict_prob_data, target_prob_data, bottom_diff);

    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.

    //const int nthreads = outer_num_ * inner_num_;
    
    // if (normalization_ == LossParameter_NormalizationMode_VALID &&
    //     has_ignore_label_) {
    //   // caffe_gpu_asum(nthreads, counts, &valid_count);
    //   caffe_gpu_asum(nthreads,  )
    // }
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, outer_num_, inner_num_, valid_count);
    const Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_gpu_scal(predict_prob_.count(), loss_weight , bottom_diff);
  }
}

// INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);
INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithCrossEntropyLossLayer);

}  // namespace caffe
