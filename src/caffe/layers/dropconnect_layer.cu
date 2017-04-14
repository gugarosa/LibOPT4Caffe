#include <vector>

#include "caffe/layers/dropconnect_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DropconnectForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
__global__ void DropconnectScalarMultiply(const int n, const Dtype* in, const float scalar, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (1-scalar);
  }
}

template <typename Dtype>
__global__ void DropconnectScalarMultiply2(const int n, const Dtype* in, const float scalar, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (1-scalar) * scalar;
  }
}

template <typename Dtype>
__global__ void DropconnectGaussian(const int n, const Dtype* mu, const Dtype* var, Dtype* g, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    //caffe_gpu_rng_gaussian(1, mu[index], var[index], g);
    out[index] = g[0];
  }
}

template <typename Dtype>
void DropconnectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int size = bottom[0]->count();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int count = this->blobs_[0]->count();
  dropped_weight_.Reshape(this->blobs_[0]->shape());
  inference_weight_.Reshape(this->blobs_[0]->shape());
  mean_inference_.Reshape(top[0]->shape());
  var_inference_.Reshape(top[0]->shape());
  Dtype* dropped_weight = static_cast<Dtype*>(dropped_weight_.mutable_gpu_data());
  Dtype* inference_weight = static_cast<Dtype*>(inference_weight_.mutable_gpu_data());
  Dtype* mean_inference = static_cast<Dtype*>(mean_inference_.mutable_gpu_data());
  Dtype* var_inference = static_cast<Dtype*>(var_inference_.mutable_gpu_data());
  if (this->phase_ == TRAIN) {
      unsigned int* mask = static_cast<unsigned int*>(rand_mat_.mutable_gpu_data());
      caffe_gpu_rng_uniform(count, mask);
      DropconnectForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, weight, mask, uint_thres_, scale_, dropped_weight);
      CUDA_POST_KERNEL_CHECK;
      if (M_ == 1) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                             dropped_weight, bottom_data, (Dtype)0., top_data);
      } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              transpose_ ? CblasNoTrans : CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              bottom_data, dropped_weight, (Dtype)0., top_data);
      }
  } else {
      const int top_count = top[0]->count();
      DropconnectScalarMultiply<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, weight, threshold_, inference_weight);
      CUDA_POST_KERNEL_CHECK;
      squared_weight_.Reshape(this->blobs_[0]->shape());
      squared_bottom_.Reshape(bottom[0]->shape());
      partial_inference_.Reshape(top[0]->shape());
      gaussian_.Reshape(top[0]->shape());
      Dtype* squared_weight = static_cast<Dtype*>(squared_weight_.mutable_gpu_data());
      Dtype* squared_bottom = static_cast<Dtype*>(squared_bottom_.mutable_gpu_data());
      Dtype* partial_inference = static_cast<Dtype*>(partial_inference_.mutable_gpu_data());
      Dtype* gaussian = static_cast<Dtype*>(gaussian_.mutable_gpu_data());
      if (M_ == 1) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                             inference_weight, bottom_data, (Dtype)0., mean_inference);
        caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                             weight, weight, (Dtype)0., squared_weight);
        caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                             bottom_data, bottom_data, (Dtype)0., squared_bottom);
        caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                             squared_weight, squared_bottom, (Dtype)0., partial_inference);
        DropconnectScalarMultiply2<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(top_count, partial_inference, threshold_, var_inference);
        CUDA_POST_KERNEL_CHECK;
        DropconnectGaussian<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(top_count, mean_inference, var_inference, gaussian, top_data);
        CUDA_POST_KERNEL_CHECK;
      } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              transpose_ ? CblasNoTrans : CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              bottom_data, inference_weight, (Dtype)0., mean_inference);
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              transpose_ ? CblasNoTrans : CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              weight, weight, (Dtype)0., squared_weight);
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              transpose_ ? CblasNoTrans : CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              bottom_data, bottom_data, (Dtype)0., squared_bottom);
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              transpose_ ? CblasNoTrans : CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              squared_bottom, squared_weight, (Dtype)0., partial_inference);
        DropconnectScalarMultiply2<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(top_count, partial_inference, threshold_, var_inference);
        CUDA_POST_KERNEL_CHECK;
        DropconnectGaussian<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(top_count, mean_inference, var_inference, gaussian, top_data);
        CUDA_POST_KERNEL_CHECK;
      }
  }
}

template <typename Dtype>
__global__ void DropconnectBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void DropconnectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        if (this->phase_ == TRAIN) {
            const unsigned int* mask = static_cast<const unsigned int*>(rand_mat_.gpu_data());
            const int count = this->blobs_[0]->count();
            // Gradient with respect to weight
              caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                  N_, K_, M_,
                  (Dtype)1., top_diff, bottom_data,
                  (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
            DropconnectBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                   count, this->blobs_[0]->gpu_diff(), mask, uint_thres_, scale_, this->blobs_[0]->mutable_gpu_diff());
        } else {
            // Gradient with respect to weight
              caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                  N_, K_, M_,
                  (Dtype)1., top_diff, bottom_data,
                  (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
        }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    if (this->phase_ == TRAIN) {
        const unsigned int* mask = static_cast<const unsigned int*>(rand_mat_.gpu_data());
        const int count = this->blobs_[0]->count();
        //DropconnectBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
               //count, this->blobs_[0]->gpu_data(), mask, uint_thres_, scale_, this->blobs_[0]->mutable_gpu_data());
        // Gradient with respect to bottom data
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
              M_, K_, N_,
             (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
             (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
        // Gradient with respect to bottom data
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
              M_, K_, N_,
             (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
             (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropconnectLayer);

}  // namespace caffe