#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/multilabel_transformer_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

const float eps = 1e-6;
const int BATCH_SIZE = 15;
const int NUM_CLASSES = 20;
const int NUM_INSTANCE = 1;

template <typename TypeParam>
class MultilabelTransformerLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultilabelTransformerLayerTest():
  	blob_bottom_(new Blob<Dtype>(1, 1, NUM_INSTANCE, 8)),
    blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
          
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MultilabelTransformerLayerTest() {
  	delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultilabelTransformerLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultilabelTransformerLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultilabelTransformerParameter* multilabel_transformer_param =
      layer_param.mutable_multilabel_transformer_param();
  multilabel_transformer_param->set_num_classes(NUM_CLASSES);
  multilabel_transformer_param->set_batch_size(BATCH_SIZE);
  MultilabelTransformerLayer<Dtype> layer(layer_param);

  Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
  int count = 0;
  int label_idx = 1;
  for(int i = 0; i < BATCH_SIZE; ++i){
  	for(int j = 0; j < NUM_INSTANCE; ++j){
      int data_idx = count * 8;
  		bottom_data[data_idx] = i;
  		bottom_data[data_idx + 1] = i + 1;
  		// bottom_data[data_idx + 2] = j;
  		// bottom_data[idx++] = 0.5 - 1.0 / (float) (i + j + 3);
  		// bottom_data[idx++] = 0.5 - 1.0 / (float) (i + j + 4);
  		// bottom_data[idx++] = 0.5 + 1.0 / (float) (i + j + 3);
  		// bottom_data[idx++] = 0.5 + 1.0 / (float) (i + j + 4);
  		label_idx++;
      count++;
  	}
  }
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), BATCH_SIZE);
  EXPECT_EQ(this->blob_top_->channels(), NUM_CLASSES);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);

  for (int i = 0; i < BATCH_SIZE; ++i){
  	// int label_start_idx = i * NUM_INSTANCE;
  	int data_idx = i * NUM_CLASSES;
  	for (int j = 0; j < NUM_CLASSES; ++j){
  		if (j == i ){
  			EXPECT_EQ(this->blob_top_->cpu_data()[data_idx + j] , 1);
  		} else {
  			EXPECT_EQ(this->blob_top_->cpu_data()[data_idx + j] , 0);
  		}
  	}
	}
}

} //namespace caffe