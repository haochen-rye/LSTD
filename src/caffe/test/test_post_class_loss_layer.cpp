#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
// #include "caffe/layers/annotated_data_layer.hpp"
// #include "caffe/layers/conv_layer.hpp"
// #include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/post_class_loss_layer.hpp"
// #include "caffe/layers/permute_layer.hpp"
// #include "caffe/layers/roi_pooling_layer.hpp"
#include "caffe/proto/caffe.pb.h"
// #include "caffe/util/db.hpp"
// #include "caffe/util/io.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

static bool kBoolChoices[] = {true, false};

static MultiBoxLossParameter_ConfLossType kConfLossTypes[] = {
  MultiBoxLossParameter_ConfLossType_SOFTMAX,
  MultiBoxLossParameter_ConfLossType_LOGISTIC};
static MultiBoxLossParameter_MatchType kMatchTypes[] = {
  MultiBoxLossParameter_MatchType_BIPARTITE,
  MultiBoxLossParameter_MatchType_PER_PREDICTION};
static LossParameter_NormalizationMode kNormalizationModes[] = {
  LossParameter_NormalizationMode_BATCH_SIZE,
  LossParameter_NormalizationMode_FULL,
  LossParameter_NormalizationMode_VALID,
  LossParameter_NormalizationMode_NONE};
static MultiBoxLossParameter_MiningType kMiningType[] = {
  MultiBoxLossParameter_MiningType_NONE,
  MultiBoxLossParameter_MiningType_MAX_NEGATIVE,
  MultiBoxLossParameter_MiningType_HARD_EXAMPLE};

template <typename TypeParam>
class PostClassLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;


  void FillItem(Dtype* blob_data, const string values) {
    // Split values to vector of items.
    vector<string> items;
    std::istringstream iss(values);
    std::copy(std::istream_iterator<string>(iss),
              std::istream_iterator<string>(), back_inserter(items));
    int num_items = items.size();
    CHECK_EQ(num_items, 8);

    for (int i = 0; i < 8; ++i) {
      if (i >= 3 && i <= 6) {
        blob_data[i] = atof(items[i].c_str());
      } else {
        blob_data[i] = atoi(items[i].c_str());
      }
    }
  }

  void FillProposal(Dtype* blob_data, const string values) {
    // Split values to vector of items.
    vector<string> items;
    std::istringstream iss(values);
    std::copy(std::istream_iterator<string>(iss),
              std::istream_iterator<string>(), back_inserter(items));
    int num_items = items.size();
    CHECK_EQ(num_items, 7);

    for (int i = 0; i < 7; ++i) {
      if (i >= 3 && i <= 6) {
        blob_data[i] = atof(items[i].c_str());
      } else {
        blob_data[i] = atoi(items[i].c_str());
      }
    }
  }

 protected:
  PostClassLossLayerTest()
      : num_(3),
        num_classes_(3),
        num_priors_(2),
        blob_bottom_proposal_(new Blob<Dtype>( 1, 1, num_* num_priors_ , 7)),
        blob_bottom_conf_(new Blob<Dtype>(
                num_, num_priors_ * num_classes_, 1, 1)),
        blob_bottom_gt_(new Blob<Dtype>(1, 1, 4, 8)),
        blob_top_loss_(new Blob<Dtype>()) {

    // FILL THE VALUES
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_conf_);

    Dtype* proposal_data = blob_bottom_proposal_->mutable_cpu_data();
    FillProposal(proposal_data, "0 1 1 0.1 0.15 0.2 0.3");
    FillProposal(proposal_data + 7, "0 1 1 0.05 0.15 0.25 0.35");
    FillProposal(proposal_data + 7 *2, "1 1 1 0.07 0.06 0.29 0.37");
    FillProposal(proposal_data + 7 *3, "1 1 1 0.03 0.15 0.32 0.39");
    FillProposal(proposal_data + 7 *4, "2 1 1 0.14 0.05 0.27 0.27");
    FillProposal(proposal_data + 7 *5, "2 1 1 0.12 0.1 0.37 0.28");

    Dtype* gt_data = blob_bottom_gt_->mutable_cpu_data();
    FillItem(gt_data, "0 1 0 0.1 0.1 0.3 0.3 0");
    FillItem(gt_data + 8, "2 1 0 0.1 0.1 0.3 0.3 0");
    FillItem(gt_data + 8 * 2, "2 2 0 0.2 0.2 0.4 0.4 0");
    FillItem(gt_data + 8 * 3, "2 2 1 0.6 0.6 0.8 0.9 1");


    blob_bottom_vec_.push_back(blob_bottom_conf_);
    blob_bottom_vec_.push_back(blob_bottom_proposal_);
    blob_bottom_vec_.push_back(blob_bottom_gt_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~PostClassLossLayerTest() {
    delete blob_bottom_proposal_;
    delete blob_bottom_conf_;
    delete blob_bottom_gt_;
    delete blob_top_loss_;
  }

  int num_;
  int num_classes_;
  int num_priors_;
  Blob<Dtype>* blob_bottom_proposal_;
  Blob<Dtype>* blob_bottom_conf_;
  Blob<Dtype>* blob_bottom_gt_;
  Blob<Dtype>* blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PostClassLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(PostClassLossLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiBoxLossParameter* multibox_loss_param =
      layer_param.mutable_multibox_loss_param();
  multibox_loss_param->set_num_classes(3);
  bool share_location = true;
    for (int j = 0; j < 2; ++j) {
      MultiBoxLossParameter_MatchType match_type = kMatchTypes[j];
        for (int m = 0; m < 3; ++m) {
          MiningType mining_type = kMiningType[m];
          if (!share_location &&
              mining_type != MultiBoxLossParameter_MiningType_NONE) {
            continue;
          }
          multibox_loss_param->set_share_location(share_location);
          multibox_loss_param->set_match_type(match_type);
          multibox_loss_param->set_mining_type(mining_type);
          PostClassLossLayer<Dtype> layer(layer_param);
          layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        }
    }
}

TYPED_TEST(PostClassLossLayerTest, TestConfGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LossParameter* loss_param = layer_param.mutable_loss_param();
  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(false);
  MultiBoxLossParameter* multibox_loss_param =
      layer_param.mutable_multibox_loss_param();
  bool share_location = true;

  multibox_loss_param->set_num_classes(this->num_classes_);
  for (int c = 0; c < 2; ++c) {
    MultiBoxLossParameter_ConfLossType conf_loss_type = kConfLossTypes[c];
      for (int j = 0; j < 2; ++j) {
        MultiBoxLossParameter_MatchType match_type = kMatchTypes[j];
          for (int n = 0; n < 4; ++n) {
            LossParameter_NormalizationMode normalize = kNormalizationModes[n];
            loss_param->set_normalization(normalize);
            for (int u = 0; u < 2; ++u) {
              bool use_difficult_gt = kBoolChoices[u];
              for (int m = 0; m < 3; ++m) {
                MiningType mining_type = kMiningType[m];
                if (!share_location &&
                    mining_type != MultiBoxLossParameter_MiningType_NONE) {
                  continue;
                }
                multibox_loss_param->set_conf_loss_type(conf_loss_type);
                multibox_loss_param->set_share_location(share_location);
                multibox_loss_param->set_match_type(match_type);
                multibox_loss_param->set_use_difficult_gt(use_difficult_gt);
                multibox_loss_param->set_background_label_id(0);
                multibox_loss_param->set_mining_type(mining_type);
                PostClassLossLayer<Dtype> layer(layer_param);
                GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
                checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                                this->blob_top_vec_, 0);
              }
            }
          }
      }
  }
}

}  // namespace caffe
