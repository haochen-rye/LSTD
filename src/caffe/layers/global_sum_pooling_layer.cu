#include "caffe/layers/global_sum_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GlobalSumPoolingForwardGPU(const int nthreads,
          const Dtype* bottom_data, Dtype* top_data,
          const int actual_num_cls,
          const int cls_start_idx, const int inner_num,
          const int num_proposal, const int num_classes) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = 0;
    int batch_index = index / actual_num_cls;
    int label_index = index % actual_num_cls;
    for (int i = 0; i < num_proposal; i++) {
      top_data[index] += bottom_data[batch_index * inner_num + 
        i * num_classes + label_index + cls_start_idx];
    }
  }
}

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  CHECK_EQ(num_proposal_, bottom[0]->shape(1))
    << "Channels of score blob must be num_proposal_";
  CHECK_EQ(num_classes_ , bottom[0]->shape(2))
    << "Height of score must be num_classes";
  const int nthreads = batch_size_ * actual_num_cls_;
      
  GlobalSumPoolingForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, top_data, 
        actual_num_cls_, cls_start_idx_, inner_num_, 
        num_proposal_, num_classes_ );

  // const int outer_num_ = bottom[0]->shape(0);
  // const int K_ = bottom[0]->count() / outer_num_;
  // const int nthreads = K_;
  // GlobalSumPoolingForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
  //     CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, top_data, outer_num_, 
  //                               K_);
}

template <typename Dtype>
__global__ void GlobalSumPoolingBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff, 
          const int actual_num_cls, const int cls_start_idx,
          const int inner_num,const int num_proposal, 
          const int num_classes) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // compute the batch_index, proposal_index, label_index
    const int k = index % actual_num_cls;
    const int j = (index / actual_num_cls) % num_proposal;
    const int i = index / actual_num_cls / num_proposal;
    // compute the actual index corresponidng to the top index
    // the mismatch of index caused by backgound score at bottom
    int bottom_index = i * inner_num + j* num_classes + 
      k + cls_start_idx;
    bottom_diff[bottom_index] = top_diff[i * actual_num_cls + k];
  }
}

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  
  const int nthreads = batch_size_ * num_proposal_ * actual_num_cls_;
  // const int actual_inner_num = num_proposal_ * actual_num_cls_;
  GlobalSumPoolingBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff, 
          actual_num_cls_, cls_start_idx_,
          inner_num_, num_proposal_, num_classes_);

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GlobalSumPoolingLayer);
}  // namespace caffe

