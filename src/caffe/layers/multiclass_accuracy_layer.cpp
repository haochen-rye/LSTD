#include <functional>
#include <utility>

#include "caffe/layers/multiclass_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MulticlassAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    MulticlassAccuracyParameter multiclass_accuracy_param = 
      this->layer_param_.multiclass_accuracy_param();
    include_background_ = multiclass_accuracy_param.include_background();
  }

template <typename Dtype>
void MulticlassAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
  // vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(1,2,1,1);
}

template <typename Dtype>
void MulticlassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  for (int i = 0; i < batch_size_; i++){
    for (int j = 0; j < num_classes_; j++){
      int label = static_cast<int>(bottom_label[i * num_classes_ + j]);
      int score_idx = i * channels_ + j + score_start_idx_;
      if (label > 0) {
      // Update Positive accuracy and count
        true_positive += (bottom_data[score_idx] >= 0);
        false_negative += (bottom_data[score_idx] < 0);
        count_pos++;
      }
      if (label < 0) {
      // Update Negative accuracy and count
        true_negative += (bottom_data[score_idx] < 0);
        false_positive += (bottom_data[score_idx] >= 0);
        count_neg++;
      }      
    }
  }

  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype precission = (true_positive > 0)?
    (true_positive / (true_positive + false_positive)) : 0;
  // Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  // Dtype harmmean = ((count_pos + count_neg) > 0)?
  //   2 / (count_pos / true_positive + count_neg / true_negative) : 0;
  // Dtype f1_score = (true_positive > 0)?
  //   2 * true_positive /
  //   (2 * true_positive + false_positive + false_negative) : 0;

  top[0]->mutable_cpu_data()[0] = sensitivity;
  top[0]->mutable_cpu_data()[1] = precission;


  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MulticlassAccuracyLayer);
REGISTER_LAYER_CLASS(MulticlassAccuracy);

}  // namespace caffe
