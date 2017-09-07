#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/layers/multilabel_transformer_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultilabelTransformerLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  MultilabelTransformerParameter multilabel_transformer_param = 
    this->layer_param_.multilabel_transformer_param();

  batch_size_ = multilabel_transformer_param.batch_size();
  num_classes_ = multilabel_transformer_param.num_classes();

  top[0]->Reshape(1,1,batch_size_,num_classes_);
}

template <typename Dtype>
void MultilabelTransformerLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MultilabelTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(3), 8)
    << "The input label blob should habe the standard format";
  CHECK_EQ(bottom[0]->shape(1), 1)
    << "The data and label should have the num 1";
  CHECK_EQ(bottom[0]->shape(0), 1)
    << "The data and label should have the channels 1";
  // bottom_shape = bottom[0]->shape;

  // LOG(INFO) << "label_output_shape:" << bottom[0]->shape(0)
  //   << bottom[0]->shape(1) << bottom[0]->shape(2) << bottom[0]->shape(3);

  const Dtype* bottom_label = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0) , top_data);

  const int num_instance = bottom[0]->shape(2);
  for (int i = 0; i < num_instance; ++i){
    const Dtype* label_data = bottom_label + i * 8;
    const int img_id = label_data[0];
    const int label = label_data[1];
    CHECK_LE(img_id, batch_size_);
    top_data[top[0]->offset(img_id, label)] = 1;
    LOG(INFO) << "img_id: " << img_id 
      << "\tlabel: " << label << "\t"
      << top_data[top[0]->offset(img_id, label)];

  }

}

INSTANTIATE_CLASS(MultilabelTransformerLayer);
REGISTER_LAYER_CLASS(MultilabelTransformer);

}  // namespace caffe
