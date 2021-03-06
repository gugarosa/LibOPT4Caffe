#ifndef CAFFE_DROPCONNECT_LAYER_HPP_
#define CAFFE_DROPCONNECT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class DropconnectLayer : public Layer<Dtype> {
 public:
  explicit DropconnectLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Dropconnect"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool transpose_;
  Blob<unsigned int> rand_mat_;
  Blob<Dtype> dropped_weight_;
  Blob<Dtype> dropped_top_diff_;
  Blob<Dtype> inference_weight_;
  Blob<Dtype> mean_inference_;
  Blob<Dtype> squared_weight_;
  Blob<Dtype> squared_bottom_;
  Blob<Dtype> partial_inference_;
  Blob<Dtype> var_inference_;
  Blob<Dtype> gaussian_;
  Dtype threshold_;
  Dtype scale_;
  unsigned int uint_thres_;
};

}  // namespace caffe

#endif  // CAFFE_DROPCONNECT_LAYER_HPP_
