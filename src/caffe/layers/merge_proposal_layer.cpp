#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/merge_proposal_layer.hpp"

namespace caffe {

template <typename Dtype>
void MergeProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const MergeProposalParameter merge_proposal_param = 
    this->layer_param_.merge_proposal_param();
  num_rpn_ = merge_proposal_param.num_rpn();
  num_proposal_ = merge_proposal_param.num_proposal();
  batch_size_ = merge_proposal_param.batch_size();

}

template <typename Dtype>
void MergeProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < num_rpn_; ++i){
    CHECK_EQ(bottom[i]->shape(0), 1);
    CHECK_EQ(bottom[i]->shape(0), 1);
    CHECK_EQ(bottom[i]->shape(2), num_proposal_ * batch_size_);
    CHECK_EQ(bottom[i]->shape(3), 7);
  }
  vector<int> top_shape(2, 1);
  top_shape.push_back( num_rpn_ * num_proposal_ * batch_size_);
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);

}

template <typename Dtype>
void MergeProposalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int total_proposal = num_rpn_ * num_proposal_;
  Dtype* top_data = top[0]->mutable_cpu_data();
  // copy the proposal by every image
  for (int i=0; i < batch_size_; ++i){
    for (int j=0; j < num_rpn_; ++j){
      const Dtype* proposal_data = bottom[j]->cpu_data();
      // for (int k=0; k < num_proposal_; ++k){
      //   CHECK_EQ(i, proposal_data[i * num_proposal_ * 7]);
      // }
      caffe_copy(num_proposal_ * 7, proposal_data + i * num_proposal_ * 7,
        top_data + i * total_proposal * 7 + j * num_proposal_ * 7);
    }
  }
  
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(MergeProposalLayer, Forward);
#endif

INSTANTIATE_CLASS(MergeProposalLayer);
REGISTER_LAYER_CLASS(MergeProposal);

}