#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/post_class_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void PostClassOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const DetectionOutputParameter& detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
  num_classes_ = detection_output_param.num_classes();
  share_location_ = detection_output_param.share_location();
  num_loc_classes_ = share_location_ ? 1 : num_classes_;
  background_label_id_ = detection_output_param.background_label_id();
  keep_top_k_ = detection_output_param.keep_top_k();
  confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
      detection_output_param.confidence_threshold() : -FLT_MAX;

  // Parameters used in nms.
  nms_threshold_ = detection_output_param.nms_param().nms_threshold();
  CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
  eta_ = detection_output_param.nms_param().eta();
  CHECK_GT(eta_, 0.);
  CHECK_LE(eta_, 1.);
  top_k_ = -1;
  if (detection_output_param.nms_param().has_top_k()) {
    top_k_ = detection_output_param.nms_param().top_k();
  }
}

template <typename Dtype>
void PostClassOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int batch_size =bottom[1]->shape(0);
  num_priors_ = bottom[0]->shape(2) / batch_size;
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  // int total_rois = keep_top_k_ * bottom[0]->shape(0);
  top_shape.push_back(1);  
  // Each row is a 7 dimension vector, which stores
  // [image_id, label, confidence, xmin, ymin, xmax, ymax]
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PostClassOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const int num = bottom[1]->num();

  // Retrieve all detection results.
  LabelBBox all_detections;
  GetDetectionResults(det_data, bottom[0]->height(), background_label_id_,
                      &all_detections);

  // Retrieve all confidences.
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                      &all_conf_scores);

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    const vector<NormalizedBBox>& img_detections = all_detections.find(i)->second;
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    map<int, vector<int> > indices;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for label " << c;
      }
      // LOG(INFO) << "APPLYING NMS FOR LABEL" << c;
      const vector<float>& scores = conf_scores.find(c)->second;

// // SEE THE SCORE RESULT FOR EACH LABEL
//       ostringstream score_stream;
//       score_stream <<"THE SCORE FOR LABLE:"  <<  c  << "\tthe size:"  <<  scores.size()<<"\n" ;
//       for (int i = 0; i < scores.size();  ++i){
//           score_stream  << scores[i]  <<" ";
//       }
//       LOG(INFO) << score_stream.str();

      int label = share_location_ ? -1 : c;
      if (img_detections.size() == 0 ) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
        continue;
      }
      ApplyNMSFast(img_detections, scores, confidence_threshold_, nms_threshold_, eta_,
          top_k_, &(indices[c]));
      num_det += indices[c].size();
      // LOG(INFO) << "THE LABEL:" << c << "\tdetect_size:" << indices[c].size();
    }
    // LOG(INFO) << "THE NUM_DET:" << num_det  << "keep_top_k" << keep_top_k_;
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      score_index_pairs.clear();
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        // LOG(INFO) << "THE KEPT LABEL" << label;
        const vector<int>& label_indices = it->second;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened for current label.
          LOG(FATAL) << "Could not find location predictions for " << label;
          continue;
        }
        const vector<float>& scores = conf_scores.find(label)->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          CHECK_LT(idx, scores.size());
          score_index_pairs.push_back(std::make_pair(
                  scores[idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        // LOG(INFO) << "THE STORE NEW LABEL"  <<  label;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);
  Dtype* top_data;
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = num;
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += 7;
    }
  } else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
  }

  int count = 0;
  for (int i = 0; i < num; ++i) {
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    const vector<NormalizedBBox>& img_detections = all_detections.find(i)->second;   
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for " << label;
        continue;
      }
      const vector<float>& scores = conf_scores.find(label)->second;
      int loc_label = share_location_ ? -1 : label;
      if ( img_detections.size() == 0 ) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for " << loc_label;
        continue;
      }
      vector<int>& indices = it->second;
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * 7] = i;
        top_data[count * 7 + 1] = label;
        top_data[count * 7 + 2] = scores[idx];
        const NormalizedBBox& bbox = img_detections[idx];
        top_data[count * 7 + 3] = bbox.xmin();
        top_data[count * 7 + 4] = bbox.ymin();
        top_data[count * 7 + 5] = bbox.xmax();
        top_data[count * 7 + 6] = bbox.ymax();
        ++count;
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(PostClassOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(PostClassOutputLayer);
REGISTER_LAYER_CLASS(PostClassOutput);

}


// #include <algorithm>
// #include <fstream>  // NOLINT(readability/streams)
// #include <map>
// #include <string>
// #include <utility>
// #include <vector>

// #include "boost/filesystem.hpp"
// #include "boost/foreach.hpp"

// #include "caffe/layers/post_class_output_layer.hpp"

// namespace caffe {

// template <typename Dtype>
// void PostClassOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {
//   const DetectionOutputParameter& detection_output_param =
//       this->layer_param_.detection_output_param();
//   CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
//   num_classes_ = detection_output_param.num_classes();
//   share_location_ = detection_output_param.share_location();
//   num_loc_classes_ = share_location_ ? 1 : num_classes_;
//   background_label_id_ = detection_output_param.background_label_id();
//   keep_top_k_ = detection_output_param.keep_top_k();
//   confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
//       detection_output_param.confidence_threshold() : -FLT_MAX;

//   // Parameters used in nms.
//   nms_threshold_ = detection_output_param.nms_param().nms_threshold();
//   CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
//   eta_ = detection_output_param.nms_param().eta();
//   CHECK_GT(eta_, 0.);
//   CHECK_LE(eta_, 1.);
//   top_k_ = -1;
//   if (detection_output_param.nms_param().has_top_k()) {
//     top_k_ = detection_output_param.nms_param().top_k();
//   }

//   const SaveOutputParameter& save_output_param =
//       detection_output_param.save_output_param();
//   output_directory_ = save_output_param.output_directory();
//   if (!output_directory_.empty()) {
//     if (boost::filesystem::is_directory(output_directory_)) {
//       boost::filesystem::remove_all(output_directory_);
//     }
//     if (!boost::filesystem::create_directories(output_directory_)) {
//         LOG(WARNING) << "Failed to create directory: " << output_directory_;
//     }
//   }
//   output_name_prefix_ = save_output_param.output_name_prefix();
//   need_save_ = output_directory_ == "" ? false : true;
//   output_format_ = save_output_param.output_format();
//   if (save_output_param.has_label_map_file()) {
//     string label_map_file = save_output_param.label_map_file();
//     if (label_map_file.empty()) {
//       // Ignore saving if there is no label_map_file provided.
//       LOG(WARNING) << "Provide label_map_file if output results to files.";
//       need_save_ = false;
//     } else {
//       LabelMap label_map;
//       CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
//           << "Failed to read label map file: " << label_map_file;
//       CHECK(MapLabelToName(label_map, true, &label_to_name_))
//           << "Failed to convert label to name.";
//       CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
//           << "Failed to convert label to display name.";
//     }
//   } else {
//     need_save_ = false;
//   }
//   if (save_output_param.has_name_size_file()) {
//     string name_size_file = save_output_param.name_size_file();
//     if (name_size_file.empty()) {
//       // Ignore saving if there is no name_size_file provided.
//       LOG(WARNING) << "Provide name_size_file if output results to files.";
//       need_save_ = false;
//     } else {
//       std::ifstream infile(name_size_file.c_str());
//       CHECK(infile.good())
//           << "Failed to open name size file: " << name_size_file;
//       // The file is in the following format:
//       //    name height width
//       //    ...
//       string name;
//       int height, width;
//       while (infile >> name >> height >> width) {
//         names_.push_back(name);
//         sizes_.push_back(std::make_pair(height, width));
//       }
//       infile.close();
//       if (save_output_param.has_num_test_image()) {
//         num_test_image_ = save_output_param.num_test_image();
//       } else {
//         num_test_image_ = names_.size();
//       }
//       CHECK_LE(num_test_image_, names_.size());
//     }
//   } else {
//     need_save_ = false;
//   }
//   has_resize_ = save_output_param.has_resize_param();
//   if (has_resize_) {
//     resize_param_ = save_output_param.resize_param();
//   }
//   name_count_ = 0;
  
// }

// template <typename Dtype>
// void PostClassOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {

//   if (need_save_) {
//     CHECK_LE(name_count_, names_.size());
//     if (name_count_ % num_test_image_ == 0) {
//       // Clean all outputs.
//       if (output_format_ == "VOC") {
//         boost::filesystem::path output_directory(output_directory_);
//         for (map<int, string>::iterator it = label_to_name_.begin();
//              it != label_to_name_.end(); ++it) {
//           if (it->first == background_label_id_) {
//             continue;
//           }
//           std::ofstream outfile;
//           boost::filesystem::path file(
//               output_name_prefix_ + it->second + ".txt");
//           boost::filesystem::path out_file = output_directory / file;
//           outfile.open(out_file.string().c_str(), std::ofstream::out);
//         }
//       }
//     }
//   }

//   int batch_size =bottom[1]->shape(0);
//   num_priors_ = bottom[0]->shape(2) / batch_size;
//   CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
//       << "Number of priors must match number of confidence predictions.";
//   // num() and channels() are 1.
//   vector<int> top_shape(2, 1);
//   // Since the number of bboxes to be kept is unknown before nms, we manually
//   // set it to (fake) 1.
//   // int total_rois = keep_top_k_ * bottom[0]->shape(0);
//   top_shape.push_back(1);  
//   // Each row is a 7 dimension vector, which stores
//   // [image_id, label, confidence, xmin, ymin, xmax, ymax]
//   top_shape.push_back(7);
//   top[0]->Reshape(top_shape);
// }

// template <typename Dtype>
// void PostClassOutputLayer<Dtype>::Forward_cpu(
//     const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//   const Dtype* det_data = bottom[0]->cpu_data();
//   const Dtype* conf_data = bottom[1]->cpu_data();
//   const int num = bottom[1]->num();

//   // Retrieve all detection results.
//   LabelBBox all_detections;
//   GetDetectionResults(det_data, bottom[0]->height(), background_label_id_,
//                       &all_detections);

//   // Retrieve all confidences.
//   vector<map<int, vector<float> > > all_conf_scores;
//   GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
//                       &all_conf_scores);

//   int num_kept = 0;
//   vector<map<int, vector<int> > > all_indices;
//   for (int i = 0; i < num; ++i) {
//     const vector<NormalizedBBox>& img_detections = all_detections.find(i)->second;
//     const map<int, vector<float> >& conf_scores = all_conf_scores[i];
//     map<int, vector<int> > indices;
//     int num_det = 0;
//     for (int c = 0; c < num_classes_; ++c) {
//       if (c == background_label_id_) {
//         // Ignore background class.
//         continue;
//       }
//       if (conf_scores.find(c) == conf_scores.end()) {
//         // Something bad happened if there are no predictions for current label.
//         LOG(FATAL) << "Could not find confidence predictions for label " << c;
//       }
//       // LOG(INFO) << "APPLYING NMS FOR LABEL" << c;
//       const vector<float>& scores = conf_scores.find(c)->second;

// // // SEE THE SCORE RESULT FOR EACH LABEL
// //       ostringstream score_stream;
// //       score_stream <<"THE SCORE FOR LABLE:"  <<  c  << "\tthe size:"  <<  scores.size()<<"\n" ;
// //       for (int i = 0; i < scores.size();  ++i){
// //           score_stream  << scores[i]  <<" ";
// //       }
// //       LOG(INFO) << score_stream.str();

//       int label = share_location_ ? -1 : c;
//       if (img_detections.size() == 0 ) {
//         // Something bad happened if there are no predictions for current label.
//         LOG(FATAL) << "Could not find location predictions for label " << label;
//         continue;
//       }
//       ApplyNMSFast(img_detections, scores, confidence_threshold_, nms_threshold_, eta_,
//           top_k_, &(indices[c]));
//       num_det += indices[c].size();
//       // LOG(INFO) << "THE LABEL:" << c << "\tdetect_size:" << indices[c].size();
//     }
//     // LOG(INFO) << "THE NUM_DET:" << num_det  << "keep_top_k" << keep_top_k_;
//     if (keep_top_k_ > -1 && num_det > keep_top_k_) {
//       vector<pair<float, pair<int, int> > > score_index_pairs;
//       score_index_pairs.clear();
//       for (map<int, vector<int> >::iterator it = indices.begin();
//            it != indices.end(); ++it) {
//         int label = it->first;
//         // LOG(INFO) << "THE KEPT LABEL" << label;
//         const vector<int>& label_indices = it->second;
//         if (conf_scores.find(label) == conf_scores.end()) {
//           // Something bad happened for current label.
//           LOG(FATAL) << "Could not find location predictions for " << label;
//           continue;
//         }
//         const vector<float>& scores = conf_scores.find(label)->second;
//         for (int j = 0; j < label_indices.size(); ++j) {
//           int idx = label_indices[j];
//           CHECK_LT(idx, scores.size());
//           score_index_pairs.push_back(std::make_pair(
//                   scores[idx], std::make_pair(label, idx)));
//         }
//       }
//       // Keep top k results per image.
//       std::sort(score_index_pairs.begin(), score_index_pairs.end(),
//                 SortScorePairDescend<pair<int, int> >);
//       score_index_pairs.resize(keep_top_k_);
//       // Store the new indices.
//       map<int, vector<int> > new_indices;
//       for (int j = 0; j < score_index_pairs.size(); ++j) {
//         int label = score_index_pairs[j].second.first;
//         int idx = score_index_pairs[j].second.second;
//         // LOG(INFO) << "THE STORE NEW LABEL"  <<  label;
//         new_indices[label].push_back(idx);
//       }
//       all_indices.push_back(new_indices);
//       num_kept += keep_top_k_;
//     } else {
//       all_indices.push_back(indices);
//       num_kept += num_det;
//     }
//   }

//   vector<int> top_shape(2, 1);
//   top_shape.push_back(num_kept);
//   top_shape.push_back(7);
//   Dtype* top_data;
//   if (num_kept == 0) {
//     LOG(INFO) << "Couldn't find any detections";
//     top_shape[2] = num;
//     top[0]->Reshape(top_shape);
//     top_data = top[0]->mutable_cpu_data();
//     caffe_set<Dtype>(top[0]->count(), -1, top_data);
//     // Generate fake results per image.
//     for (int i = 0; i < num; ++i) {
//       top_data[0] = i;
//       top_data += 7;
//     }
//   } else {
//     top[0]->Reshape(top_shape);
//     top_data = top[0]->mutable_cpu_data();
//   }

//   int count = 0;
//   for (int i = 0; i < num; ++i) {
//     const map<int, vector<float> >& conf_scores = all_conf_scores[i];
//     const vector<NormalizedBBox>& img_detections = all_detections.find(i)->second;   
//     for (map<int, vector<int> >::iterator it = all_indices[i].begin();
//          it != all_indices[i].end(); ++it) {
//       int label = it->first;
//       if (conf_scores.find(label) == conf_scores.end()) {
//         // Something bad happened if there are no predictions for current label.
//         LOG(FATAL) << "Could not find confidence predictions for " << label;
//         continue;
//       }
//       const vector<float>& scores = conf_scores.find(label)->second;
//       int loc_label = share_location_ ? -1 : label;
//       if ( img_detections.size() == 0 ) {
//         // Something bad happened if there are no predictions for current label.
//         LOG(FATAL) << "Could not find location predictions for " << loc_label;
//         continue;
//       }
//       vector<int>& indices = it->second;
//       if (need_save_) {
//         CHECK(label_to_name_.find(label) != label_to_name_.end())
//           << "Cannot find label: " << label << " in the label map.";
//         CHECK_LT(name_count_, names_.size());
//       }
//       for (int j = 0; j < indices.size(); ++j) {
//         int idx = indices[j];
//         top_data[count * 7] = i;
//         top_data[count * 7 + 1] = label;
//         top_data[count * 7 + 2] = scores[idx];
//         const NormalizedBBox& bbox = img_detections[idx];
//         top_data[count * 7 + 3] = bbox.xmin();
//         top_data[count * 7 + 4] = bbox.ymin();
//         top_data[count * 7 + 5] = bbox.xmax();
//         top_data[count * 7 + 6] = bbox.ymax();
//         if (need_save_) {
//           NormalizedBBox out_bbox;
//           OutputBBox(bbox, sizes_[name_count_], has_resize_, resize_param_,
//                      &out_bbox);
//           float score = top_data[count * 7 + 2];
//           float xmin = out_bbox.xmin();
//           float ymin = out_bbox.ymin();
//           float xmax = out_bbox.xmax();
//           float ymax = out_bbox.ymax();
//           ptree pt_xmin, pt_ymin, pt_width, pt_height;
//           pt_xmin.put<float>("", round(xmin * 100) / 100.);
//           pt_ymin.put<float>("", round(ymin * 100) / 100.);
//           pt_width.put<float>("", round((xmax - xmin) * 100) / 100.);
//           pt_height.put<float>("", round((ymax - ymin) * 100) / 100.);

//           ptree cur_bbox;
//           cur_bbox.push_back(std::make_pair("", pt_xmin));
//           cur_bbox.push_back(std::make_pair("", pt_ymin));
//           cur_bbox.push_back(std::make_pair("", pt_width));
//           cur_bbox.push_back(std::make_pair("", pt_height));

//           ptree cur_det;
//           cur_det.put("image_id", names_[name_count_]);
//           if (output_format_ == "ILSVRC") {
//             cur_det.put<int>("category_id", label);
//           } else {
//             cur_det.put("category_id", label_to_name_[label].c_str());
//           }
//           cur_det.add_child("bbox", cur_bbox);
//           cur_det.put<float>("score", score);

//           detections_.push_back(std::make_pair("", cur_det));
//         }
//         ++count;
//       }
//     }
//     if (need_save_) {
//       ++name_count_;
//       if (name_count_ % num_test_image_ == 0) {
//         if (output_format_ == "VOC") {
//           map<string, std::ofstream*> outfiles;
//           for (int c = 0; c < num_classes_; ++c) {
//             if (c == background_label_id_) {
//               continue;
//             }
//             string label_name = label_to_name_[c];
//             boost::filesystem::path file(
//                 output_name_prefix_ + label_name + ".txt");
//             boost::filesystem::path out_file = output_directory / file;
//             outfiles[label_name] = new std::ofstream(out_file.string().c_str(),
//                 std::ofstream::out);
//           }
//           BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
//             ptree pt = det.second;
//             string label_name = pt.get<string>("category_id");
//             if (outfiles.find(label_name) == outfiles.end()) {
//               std::cout << "Cannot find " << label_name << std::endl;
//               continue;
//             }
//             string image_name = pt.get<string>("image_id");
//             float score = pt.get<float>("score");
//             vector<int> bbox;
//             BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
//               bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
//             }
//             *(outfiles[label_name]) << image_name;
//             *(outfiles[label_name]) << " " << score;
//             *(outfiles[label_name]) << " " << bbox[0] << " " << bbox[1];
//             *(outfiles[label_name]) << " " << bbox[0] + bbox[2];
//             *(outfiles[label_name]) << " " << bbox[1] + bbox[3];
//             *(outfiles[label_name]) << std::endl;
//           }
//           for (int c = 0; c < num_classes_; ++c) {
//             if (c == background_label_id_) {
//               continue;
//             }
//             string label_name = label_to_name_[c];
//             outfiles[label_name]->flush();
//             outfiles[label_name]->close();
//             delete outfiles[label_name];
//           }
//         } else if (output_format_ == "COCO") {
//           boost::filesystem::path output_directory(output_directory_);
//           boost::filesystem::path file(output_name_prefix_ + ".json");
//           boost::filesystem::path out_file = output_directory / file;
//           std::ofstream outfile;
//           outfile.open(out_file.string().c_str(), std::ofstream::out);

//           boost::regex exp("\"(null|true|false|-?[0-9]+(\\.[0-9]+)?)\"");
//           ptree output;
//           output.add_child("detections", detections_);
//           std::stringstream ss;
//           write_json(ss, output);
//           std::string rv = boost::regex_replace(ss.str(), exp, "$1");
//           outfile << rv.substr(rv.find("["), rv.rfind("]") - rv.find("["))
//               << std::endl << "]" << std::endl;
//         } else if (output_format_ == "ILSVRC") {
//           boost::filesystem::path output_directory(output_directory_);
//           boost::filesystem::path file(output_name_prefix_ + ".txt");
//           boost::filesystem::path out_file = output_directory / file;
//           std::ofstream outfile;
//           outfile.open(out_file.string().c_str(), std::ofstream::out);

//           BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
//             ptree pt = det.second;
//             int label = pt.get<int>("category_id");
//             string image_name = pt.get<string>("image_id");
//             float score = pt.get<float>("score");
//             vector<int> bbox;
//             BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
//               bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
//             }
//             outfile << image_name << " " << label << " " << score;
//             outfile << " " << bbox[0] << " " << bbox[1];
//             outfile << " " << bbox[0] + bbox[2];
//             outfile << " " << bbox[1] + bbox[3];
//             outfile << std::endl;
//           }
//         }
//         name_count_ = 0;
//         detections_.clear();
//       }
//     }

//   }
// }


// #ifdef CPU_ONLY
// STUB_GPU_FORWARD(PostClassOutputLayer, Forward);
// #endif

// INSTANTIATE_CLASS(PostClassOutputLayer);
// REGISTER_LAYER_CLASS(PostClassOutput);

// }
