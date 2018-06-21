
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/global_sum_pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

const int BATCH_SIZE = 2;
const int NUM_PROPOSAL = 3;
const int NUM_CLASSES = 6;

template <typename TypeParam>
class GlobalSumPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GlobalSumPoolingLayerTest():
      // : blob_bottom_(new Blob<Dtype>(2, 3, 6, 1)),
        blob_bottom_(new Blob<Dtype>(BATCH_SIZE, NUM_PROPOSAL, NUM_CLASSES, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GlobalSumPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GlobalSumPoolingLayerTest, TestDtypesAndDevices);

// only test when background is included
TYPED_TEST(GlobalSumPoolingLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GlobalSumPoolingParameter* global_sum_pooling_param =
      layer_param.mutable_global_sum_pooling_param();
  // const int num_classes = 6;
  // const int num_proposal = 3;
  // const int batch_size = 2;
  // const int actual_classes = num_classes -1;
  const int actual_classes = NUM_CLASSES - 1;
  global_sum_pooling_param->set_num_classes(NUM_CLASSES);
  global_sum_pooling_param->set_num_proposal(NUM_PROPOSAL);
  this->blob_bottom_->Reshape(BATCH_SIZE,NUM_PROPOSAL,NUM_CLASSES,1);
  for (int i = 0; i < BATCH_SIZE; ++i){
  	for (int j = 0; j < NUM_PROPOSAL; ++j){
  		for (int k = 0; k < NUM_CLASSES; ++k){
  			int data_index = (i * NUM_PROPOSAL + j) * NUM_CLASSES + k;
  			this->blob_bottom_->mutable_cpu_data()[data_index] =
         (float)k / (float)10;
  		}
  	}
  }
  GlobalSumPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), BATCH_SIZE);
  EXPECT_EQ(this->blob_top_->channels(), NUM_CLASSES - 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-6;
  // output:
  // 	the score for label k is: k * NUM_PROPOSAL / 10
  for (int i = 0; i < BATCH_SIZE; ++i){
  	for (int j = 0; j < actual_classes; ++j){
  		int top_index = i * actual_classes + j;
			  EXPECT_NEAR(this->blob_top_->cpu_data()[top_index], 
			  	(float)(j + 1) * (float)NUM_PROPOSAL / (float)10, epsilon);

  	}
  }

}


TYPED_TEST(GlobalSumPoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GlobalSumPoolingParameter* global_sum_pooling_param =
      layer_param.mutable_global_sum_pooling_param();
  global_sum_pooling_param->set_num_classes(NUM_CLASSES);
  global_sum_pooling_param->set_num_proposal(NUM_PROPOSAL);
  GlobalSumPoolingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
