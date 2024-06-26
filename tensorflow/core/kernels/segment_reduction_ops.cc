/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/segment_reduction_ops.h"
#include "tensorflow/core/kernels/segment_reduction_ops_util.h"

#include <cstdint>
#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// This operator handles reducing segments along the first dimension.
// See core/ops/math_ops.cc for more details.
template <typename Device, class T, class Index, typename Reducer,
          int default_value>
class SegmentReductionOp : public OpKernel {
 public:
  explicit SegmentReductionOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& segment_ids = context->input(1);

    if (!SegmentReductionDoValidation(context, input, segment_ids)) {
      return;
    }

    const int64 num_indices = segment_ids.NumElements();
    auto input_flat = input.flat_outer_dims<T>();
    const int64 num_col = input_flat.dimension(1);

    const auto segment_vec = segment_ids.vec<Index>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const Index output_rows =
        num_indices > 0
            ? internal::SubtleMustCopy(segment_vec(num_indices - 1)) + 1
            : 0;
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    OP_REQUIRES_OK(context, output_shape.SetDimWithStatus(0, output_rows));

    // Note that we do not initialize the output buffer with a default value, so
    // we need to explicitly set missing indices to the default value.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) return;
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::DenseIndex, 1> dims_to_reduce;
    dims_to_reduce[0] = 0;
#else
    Eigen::IndexList<Eigen::type2index<0> > dims_to_reduce;
#endif
    Index start = 0, end = 1;

    Index uninitialized_index = 0;  // Index from which the output is not set.
    Index out_index = internal::SubtleMustCopy(segment_vec(start));

    // TODO(agarwal): if this loop becomes a bottleneck, consider sharding it
    // across threads.
    Eigen::DSizes<Eigen::DenseIndex, 1> out_slice_shape(num_col);
    while (end <= num_indices) {
      // We initialize next_index to 0 to avoid "warning: 'next_index' may be
      // used uninitialized in this function" in the Mac build (since the
      // compiler isn't smart enough to realize the code is safe).
      Index next_index = 0;
      if (end < num_indices) {
        next_index = internal::SubtleMustCopy(segment_vec(end));
        if (out_index == next_index) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids are growing.
        OP_REQUIRES(context, out_index < next_index,
                    errors::InvalidArgument("segment ids are not increasing"));
      }

      // Process segment [start, end)
      const T* in_slice_ptr = &input_flat(start, 0);
      typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                               Eigen::Unaligned>
          OutT;

      OP_REQUIRES(
          context, FastBoundsCheck(out_index, output_rows),
          errors::InvalidArgument(
              "Segment id ", out_index, " out of range [0, ", output_rows,
              "), possibly because 'segment_ids' input is not sorted."));

      // If there is a gap between two indices, we need to set that gap to the
      // default value.
      if (out_index > uninitialized_index) {
        Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
            out_index - uninitialized_index, num_col);
        Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
            gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
        gap_slice.setConstant(T(default_value));
      }

      T* out_slice_ptr = &output_flat(out_index, 0);
      OutT out_slice(out_slice_ptr, out_slice_shape);
      // We don't use out_slice.device(context->eigen_device<Device>)
      // because these pieces of work are likely to be very small and
      // the context switching overhead dwarfs any benefit we get from
      // using another thread to do this work.
      if (start == end - 1) {
        typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                                 Eigen::Unaligned>
            InT;
        InT in_slice(in_slice_ptr, out_slice_shape);
        out_slice = in_slice;
      } else {
        Eigen::DSizes<Eigen::DenseIndex, 2> in_slice_shape(end - start,
                                                           num_col);
        typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                                 Eigen::Unaligned>
            InT;
        InT in_slice(in_slice_ptr, in_slice_shape);

        out_slice = in_slice.reduce(dims_to_reduce, Reducer());
      }
      if (end >= num_indices) break;
      start = end;
      ++end;
      uninitialized_index = out_index + 1;
      out_index = next_index;
    }
  }
};

#define REGISTER_CPU_KERNEL_SEGMENT(name, functor, type, index_type, \
                                    default_value)                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name(name)                                                     \
          .Device(DEVICE_CPU)                                        \
          .TypeConstraint<type>("T")                                 \
          .TypeConstraint<index_type>("Tindices"),                   \
      SegmentReductionOp<CPUDevice, type, index_type, functor, default_value>)

#define REGISTER_REAL_CPU_KERNELS(type, index_type)                            \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentSum", Eigen::internal::SumReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentMean", Eigen::internal::MeanReducer<type>, type, index_type, 0); \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentProd", Eigen::internal::ProdReducer<type>, type, index_type, 1); \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentMin", Eigen::internal::MinReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentMax", Eigen::internal::MaxReducer<type>, \
                              type, index_type, 0)

#define REGISTER_COMPLEX_CPU_KERNELS(type, index_type)                         \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentSum", Eigen::internal::SumReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentMean", Eigen::internal::MeanReducer<type>, type, index_type, 0); \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentProd", Eigen::internal::ProdReducer<type>, type, index_type, 1);

#define REGISTER_REAL_CPU_KERNELS_ALL(type) \
  REGISTER_REAL_CPU_KERNELS(type, int32);   \
  REGISTER_REAL_CPU_KERNELS(type, int64)

#define REGISTER_COMPLEX_CPU_KERNELS_ALL(type) \
  REGISTER_COMPLEX_CPU_KERNELS(type, int32);   \
  REGISTER_COMPLEX_CPU_KERNELS(type, int64)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_REAL_CPU_KERNELS_ALL);
REGISTER_COMPLEX_CPU_KERNELS_ALL(complex64);
REGISTER_COMPLEX_CPU_KERNELS_ALL(complex128);
#undef REGISTER_CPU_KERNEL_SEGMENT
#undef REGISTER_REAL_CPU_KERNELS
#undef REGISTER_COMPLEX_CPU_KERNELS
#undef REGISTER_REAL_CPU_KERNELS_ALL
#undef REGISTER_COMPLEX_CPU_KERNELS_ALL

// ____________________________________________________________________________
// Unsorted segment reduction ops.

namespace functor {

// The ReductionFunctor implementation for CPU.
template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<CPUDevice, T, Index, InitialValueF, ReductionF> {
  void operator()(OpKernelContext* ctx, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  typename TTypes<T, 2>::ConstTensor data,
                  typename TTypes<T, 2>::Tensor output) {
    auto cpu_device = ctx->eigen_cpu_device();
    output.device(cpu_device) = output.constant(InitialValueF()());
    if (data.size() == 0) {
      return;
    }

    // This functor will reduce `N` rows input to `num_segments` rows output.
    const int64 N = segment_ids.dimension(0);
    const int64 num_segments = output.dimension(0);
    const int64_t inner_dim = data.dimension(1);
    const T* data_ptr = data.data();
    T* out_ptr = output.data();
    ReductionF reduction;

    bool data_is_1D = true;
    for (int i=1; i<data.dimensions().size(); i++) {
      if(data.dimensions()[i] != 1) data_is_1D = false;
    }

    // `num_real_segment` counts the rows actually reduced from input,
    // the rows with negative segment index will be excluded.
    // It will be used for cost model.
    int64_t num_real_segment = N;
    // `num_reductions` counts the rows actually reduced in output,
    // the rows only filled with InitialValueF() will be excluded.
    int64_t num_reductions = 0;
    // `row_counter` records how many input rows will be reduced in each
    // output row, the row only fills with InitialValueF() will keep 0.
    // Length of non-zero elements is `num_reductions`.
    std::vector<Index> row_counter(num_segments, 0);
    for (int64 i = 0; i < N; ++i) {
      Index j = internal::SubtleMustCopy(segment_ids(i));
      if (j < 0) {
        --num_real_segment;
        continue;
      }
      OP_REQUIRES(ctx, FastBoundsCheck(j, num_segments),
                  errors::InvalidArgument(
                      "segment_ids", SliceDebugString(segment_ids_shape, i),
                      " = ", j, " is out of range [0, ", num_segments, ")"));
      if (row_counter[j] == 0) num_reductions++;
      row_counter[j]++;
    }

    // Nothing to reduce. All output values equal to `InitialValueF()`.
    if (num_reductions == 0) return;

    // Parallelize by `num_segments`. It's simple, efficient and safe
    // (no data dependency):
    //
    //   input   segment_ids                 num_segments  operation
    //   | a0 |  | 0 |            worker 1:  |0|           f(a0, a1)
    //   | b0 |  | 1 |            worker 2:  |1|           f(b0, b1)
    // N | c0 |  | 2 |       -->  worker 3:  |2|           f(c0)
    //   | b1 |  | 1 |
    //   | a1 |  | 0 |
    //
    // TODO(intel-tf): Balance workload in `row_counter` to make parallelism
    //                 more efficient.
    auto reductionWorker = [&](int64_t begin, int64_t end) -> void {
      for (int64_t i = 0; i < N; i++) {
        Index j = internal::SubtleMustCopy(segment_ids(i));
        // If `j` is in work scope of this worker, do the reduction.
        if (j >= begin && j < end) {
          if (data_is_1D) {
            reduction(data_ptr[i], out_ptr[j]);
          } else {
            reduction(data.template chip<0>(i), output.template chip<0>(j));
          }
        }
      }
    };

    // Reduction functors includes Sum, Max, Min, etc. Simply consider it
    // will cost 5 cycles per operation.
    const int64_t kAverTaskSize = num_real_segment / num_segments;
    const int64_t compute_cycles = 5 * inner_dim * kAverTaskSize;
    const int64_t input_bytes = sizeof(T) * inner_dim * kAverTaskSize;
    const int64_t output_bytes = sizeof(T) * inner_dim * kAverTaskSize;
    const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
    cpu_device.parallelFor(num_segments, cost, reductionWorker);
  }
};

template <typename T>
using MatrixChip = Eigen::TensorChippingOp<0l, typename TTypes<T, 2>::Matrix>;

template <typename T>
using constMatrixChip =
    Eigen::TensorChippingOp<0l, const typename TTypes<T, 2>::ConstMatrix>;

// reduction functors
template <typename T>
struct SumOp {
  void operator()(const constMatrixChip<T> data, MatrixChip<T> output) {
    output += data;
  }
  void operator()(const T &data, T &output) {
    output += data;
  }
};

template <typename T>
struct MaxOp {
  void operator()(const constMatrixChip<T> data, MatrixChip<T> output) {
    output = data.cwiseMax(output);
  }
  void operator()(const T &data, T &output) {
    output = std::max(data, output);
  }
};

template <typename T>
struct MinOp {
  void operator()(const constMatrixChip<T> data, MatrixChip<T> output) {
    output = data.cwiseMin(output);
  }
  void operator()(const T &data, T &output) {
    output = std::min(data, output);
  }
};

template <typename T>
struct ProdOp {
  void operator()(const constMatrixChip<T> data, MatrixChip<T> output) {
    output *= data;
  }
  void operator()(const T &data, T &output) {
    output *= data;
  }
};
}  // namespace functor

#define REGISTER_CPU_KERNEL_UNSORTEDSEGMENT(                           \
    name, type, index_type, initial_value_functor, reduction_functor)  \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(name)                                                       \
          .Device(DEVICE_CPU)                                          \
          .TypeConstraint<type>("T")                                   \
          .TypeConstraint<index_type>("Tindices"),                     \
      UnsortedSegmentReductionOp<                                      \
          type, index_type,                                            \
          functor::UnsortedSegmentFunctor<CPUDevice, type, index_type, \
                                          initial_value_functor,       \
                                          reduction_functor> >)

#define REGISTER_REAL_CPU_UNSORTED_KERNELS(type, index_type)                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentSum", type, index_type,  \
                                      functor::Zero<type>,                     \
                                      functor::SumOp<type>);                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMax", type, index_type,  \
                                      functor::Lowest<type>,                   \
                                      functor::MaxOp<type>);                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMin", type, index_type,  \
                                      functor::Highest<type>,                  \
                                      functor::MinOp<type>);                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentProd", type, index_type, \
                                      functor::One<type>,                      \
                                      functor::ProdOp<type>);

#define REGISTER_COMPLEX_CPU_UNSORTED_KERNELS(type, index_type)                \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentSum", type, index_type,  \
                                      functor::Zero<type>,                     \
                                      functor::SumOp<type>);                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentProd", type, index_type, \
                                      functor::One<type>,                      \
                                      functor::ProdOp<type>)

#define REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_REAL_CPU_UNSORTED_KERNELS(type, int32);   \
  REGISTER_REAL_CPU_UNSORTED_KERNELS(type, int64)

#define REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_COMPLEX_CPU_UNSORTED_KERNELS(type, int32);   \
  REGISTER_COMPLEX_CPU_UNSORTED_KERNELS(type, int64)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL);
REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(complex64);
REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(complex128);

#undef REGISTER_REAL_CPU_UNSORTED_KERNELS
#undef REGISTER_CPU_KERNEL_UNSORTEDSEGMENT
#undef REGISTER_COMPLEX_CPU_UNSORTED_KERNELS
#undef REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL
#undef REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL

// ____________________________________________________________________________
// Sparse segment reduction ops.

// Same as SegmentReductionOp but takes as input a "sparse" tensor, represented
// by two dense tensors, one containing the data, and the other containing
// indices into the data.
template <typename Device, class T>
class SparseSegmentReductionOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionOpBase(OpKernelConstruction* context,
                                        bool is_mean, bool is_sqrtn,
                                        bool has_num_segments, T default_value)
      : OpKernel(context),
        dtidx_(DataTypeToEnum<Index>::v()),
        is_mean_(is_mean),
        is_sqrtn_(is_sqrtn),
        has_num_segments_(has_num_segments),
        default_value_(default_value) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    Index output_rows = -1;
    if (has_num_segments_) {
      const Tensor& num_segments = context->input(3);

      OP_REQUIRES(
          context, num_segments.shape().dims() == 0,
          errors::InvalidArgument("num_segments should be a scalar, not shape ",
                                  num_segments.shape().DebugString()));
      output_rows = internal::SubtleMustCopy(num_segments.scalar<int32>()());
      OP_REQUIRES(context, output_rows >= 0,
                  errors::InvalidArgument("segment ids must be >= 0"));
    }

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));

    const int64 num_indices = indices.NumElements();
    OP_REQUIRES(context, num_indices == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));

    auto input_flat = input.flat_outer_dims<T>();
    const int64 num_col = input_flat.dimension(1);
    const auto indices_vec = indices.vec<Index>();
    typedef int32 OutputRow;
    const auto segment_vec = segment_ids.vec<OutputRow>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const int64 last_segment_id =
        num_indices > 0 ? segment_vec(num_indices - 1) : 0;
    int64 limit = dtidx_ == DataType::DT_INT32 ? kint32max : kint64max;

    OP_REQUIRES(
        context, last_segment_id < limit,
        errors::InvalidArgument("Last segment id must be < kintmax, got ",
                                last_segment_id, " limit ", limit));

    const OutputRow last_segment_id_plus_one =
        num_indices > 0
            ? internal::SubtleMustCopy(segment_vec(num_indices - 1)) + 1
            : 0;
    if (has_num_segments_) {
      OP_REQUIRES(
          context, output_rows >= last_segment_id_plus_one,
          errors::InvalidArgument("segment ids must be < num_segments"));
    } else {
      output_rows = last_segment_id_plus_one;
    }
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    OP_REQUIRES_OK(context, output_shape.SetDimWithStatus(0, output_rows));

    // Note that we do not initialize the output buffer with a default value, so
    // we need to explicitly set missing indices to the default value.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) {
      if (output_rows > 0) {
        output->flat_outer_dims<T>().setConstant(default_value_);
      }
      return;
    }
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

    int64 start = 0, end = 1;
    // Index from which the output is not initialized.
    OutputRow uninitialized_index = 0;
    OutputRow out_index = internal::SubtleMustCopy(segment_vec(start));

    while (true) {
      // We initialize next_index to 0 to avoid "warning: 'next_index' may be
      // used uninitialized in this function" in the Mac build (since the
      // compiler isn't smart enough to realize the code is safe).
      OutputRow next_index = 0;
      if (end < num_indices) {
        next_index = internal::SubtleMustCopy(segment_vec(end));
        if (out_index == next_index) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids are growing.
        OP_REQUIRES(context, out_index < next_index,
                    errors::InvalidArgument("segment ids are not increasing"));
      }

      OP_REQUIRES(
          context, FastBoundsCheck(out_index, output_rows),
          errors::InvalidArgument(
              "Segment id ", out_index, " out of range [0, ", output_rows,
              "), possibly because 'segment_ids' input is not sorted."));

      // If there is a gap between two indices, we need to set that gap to the
      // default value.
      if (out_index > uninitialized_index) {
        Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
            out_index - uninitialized_index, num_col);
        Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
            gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
        gap_slice.setConstant(default_value_);
      }

      auto out = output_flat.template chip<0>(out_index);
      const int bad_offset =
          Reduce(input_flat, indices_vec, start, end - start, out);
      OP_REQUIRES(context, bad_offset < 0,
                  errors::InvalidArgument(
                      "Bad: indices[", start + bad_offset,
                      "] == ", indices_vec(start + bad_offset),
                      " out of range [0, ", input_flat.dimension(0), ")"));

      start = end;
      ++end;
      uninitialized_index = out_index + 1;
      out_index = next_index;
      if (end > num_indices) break;
    }

    // Fill the gap at the end with the default value.
    if (uninitialized_index < output_rows) {
      Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
          output_rows - uninitialized_index, num_col);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
          gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
      gap_slice.setConstant(default_value_);
    }
  }

 private:
  const DataType dtidx_;
  typedef int32 Index;

  int64 Reduce(const typename TTypes<T>::ConstMatrix& input_flat,
               const typename TTypes<Index>::ConstVec& indices_vec, int64 start,
               int64 num,
               Eigen::TensorChippingOp<0, typename TTypes<T>::Matrix> out) {
#define INDEX(n, i)                               \
  const auto index##n = indices_vec(start + (i)); \
  if (!FastBoundsCheck(index##n, input_flat.dimension(0))) return (i);

#define L(n) input_flat.template chip<0>(index##n)

    if (num == 1) {
      INDEX(0, 0);
      out = L(0);
    } else {
      int64 r = num % 8;
      T m(1);
      if (is_mean_ && (num < 10)) {
        m = T(num);
      }
      if (is_sqrtn_ && (num < 10)) {
        m = T(sqrt(num));
      }
      switch (r) {
        case 2: {
          INDEX(0, 0);
          INDEX(1, 1);
          out = (L(0) + L(1)) / m;
          break;
        }
        case 3: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          out = (L(0) + L(1) + L(2)) / m;
          break;
        }
        case 4: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          out = (L(0) + L(1) + L(2) + L(3)) / m;
          break;
        }
        case 5: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          out = (L(0) + L(1) + L(2) + L(3) + L(4)) / m;
          break;
        }
        case 6: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5)) / m;
          break;
        }
        case 7: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6)) / m;
          break;
        }
        case 0: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          INDEX(7, 7);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7)) / m;
          r = 8;
          break;
        }
        case 1: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          INDEX(7, 7);
          INDEX(8, 8);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7) + L(8)) /
                m;
          r = 9;
          break;
        }
      }
      for (; r < num; r += 8) {
        INDEX(0, r);
        INDEX(1, r + 1);
        INDEX(2, r + 2);
        INDEX(3, r + 3);
        INDEX(4, r + 4);
        INDEX(5, r + 5);
        INDEX(6, r + 6);
        INDEX(7, r + 7);
        out += L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7);
      }
      if (is_mean_ && num >= 10) {
        out = out / static_cast<T>(num);
      }
      if (is_sqrtn_ && num >= 10) {
        out = out / static_cast<T>(sqrt(num));
      }
    }

    return -1;
#undef L
#undef INDEX
  }

  const bool is_mean_;
  const bool is_sqrtn_;
  const bool has_num_segments_;
  const T default_value_;
};

template <typename Device, class T>
class SparseSegmentReductionMeanOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionMeanWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionMeanWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionSqrtNOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSqrtNOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionSqrtNWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSqrtNWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionSumOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSumOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T>
class SparseSegmentReductionSumWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSumWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

/* ===== KERNEL registering COMMENTED, optimized Op kernels would be used. =====
#define REGISTER_CPU_SPARSE_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSum")                       \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<int32>("Tidx"),            \
                          SparseSegmentReductionSumOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseSegmentSumWithNumSegments")                            \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int32>("Tidx"),                                \
      SparseSegmentReductionSumWithNumSegmentsOp<CPUDevice, type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMean")                       \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<type>("T")                  \
                              .TypeConstraint<int32>("Tidx"),             \
                          SparseSegmentReductionMeanOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseSegmentMeanWithNumSegments")                            \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int32>("Tidx"),                                 \
      SparseSegmentReductionMeanWithNumSegmentsOp<CPUDevice, type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtN")                       \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int32>("Tidx"),              \
                          SparseSegmentReductionSqrtNOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SparseSegmentSqrtNWithNumSegments")                            \
          .Device(DEVICE_CPU)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tidx"),                                  \
      SparseSegmentReductionSqrtNWithNumSegmentsOp<CPUDevice, type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS
================= End of KERNEL registering COMMENTED block. ================ */

template <class T>
class SparseSegmentGradOpBase : public OpKernel {
 public:
  explicit SparseSegmentGradOpBase(OpKernelConstruction* context, bool is_sqrtn)
      : OpKernel(context), is_sqrtn_(is_sqrtn) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);
    const Tensor& output_dim0 = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));
    OP_REQUIRES(context, IsLegacyScalar(output_dim0.shape()),
                errors::InvalidArgument("output_dim0 should be a scalar."));

    const int64 N = indices.NumElements();
    OP_REQUIRES(context, N == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));
    typedef int32 SegmentId;
    const SegmentId M =
        internal::SubtleMustCopy(output_dim0.scalar<SegmentId>()());

    auto input_flat = input.flat_outer_dims<T>();
    typedef int32 Index;
    const auto indices_vec = indices.vec<Index>();
    const auto segment_vec = segment_ids.vec<SegmentId>();

    TensorShape output_shape = input.shape();
    OP_REQUIRES_OK(context, output_shape.SetDimWithStatus(0, M));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (M == 0 || N == 0) return;

    // Note that similar to SparseSegmentMean, we assume that segment_vec is
    // already sorted and has non-negative values.
    const SegmentId num_segments = input.dim_size(0);
    const SegmentId last_segment_id_plus_one =
        internal::SubtleMustCopy(segment_vec(N - 1)) + 1;
    OP_REQUIRES(context, last_segment_id_plus_one <= num_segments,
                errors::InvalidArgument("Invalid number of segments"));

    // Compute scaling factors for input.
    std::vector<double> scaling(num_segments, 0.0);
    for (int64 i = 0; i < N; ++i) {
      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));
      scaling[idx] += 1;
    }
    for (size_t i = 0; i < scaling.size(); ++i) {
      if (is_sqrtn_) {
        scaling[i] = 1.0 / sqrt(std::max(scaling[i], 1.0));
      } else {
        scaling[i] = 1.0 / std::max(scaling[i], 1.0);
      }
    }

    auto output_flat = output->flat_outer_dims<T>();
    output_flat.setZero();
    std::vector<bool> is_modified(M, false);

    for (int64 i = 0; i < N; ++i) {
      const Index output_idx = internal::SubtleMustCopy(indices_vec(i));
      OP_REQUIRES(context, FastBoundsCheck(output_idx, M),
                  errors::InvalidArgument("Index ", output_idx,
                                          " out of range [0, ", M, ")."));

      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));

      const T scale = static_cast<T>(scaling[idx]);
      if (is_modified[output_idx]) {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx) * scale;
        }
      } else {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx) * scale;
        }
      }
      is_modified[output_idx] = true;
    }
  }

 private:
  const bool is_sqrtn_;
};

template <class T>
class SparseSegmentMeanGradOp : public SparseSegmentGradOpBase<T> {
 public:
  explicit SparseSegmentMeanGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<T>(context, false /*is_sqrtn*/) {}
};

template <class T>
class SparseSegmentSqrtNGradOp : public SparseSegmentGradOpBase<T> {
 public:
  explicit SparseSegmentSqrtNGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<T>(context, true /*is_sqrtn*/) {}
};

/* ===== KERNEL registering COMMENTED, optimized Op kernels would be used. =====
#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMeanGrad")       \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentMeanGradOp<type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtNGrad")      \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentSqrtNGradOp<type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS
================= End of KERNEL registering COMMENTED block. ================ */
}  // namespace tensorflow
