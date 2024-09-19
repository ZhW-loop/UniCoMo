/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/tir/transform.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

/*! \brief Type for multi-dimensional index */
using MultiIndex = std::vector<PrimExpr>;
/*! \brief Vector of int64_t */
using IntVec = std::vector<int64_t>;
/*! \brief Vector of for loops */
using ForVec = std::vector<const ForNode*>;

/*!
 * \brief An unordered_map for (for, buffer) => V
 * \tparam V The value type
 */
template <class V>
using ForBufferMap = std::unordered_map<const ForNode*, std::unordered_map<const BufferNode*, V>>;

/*! \brief Given x, compute log2(|x| + 1) */
inline double slog(double x) { return x >= 0 ? std::log2(x + 1) : std::log2(-x + 1); }

namespace utils {
namespace TN    {

/*!
 * \brief Get the shape of the buffer
 * \param buffer The buffer
 * \param analyzer The analyzer
 * \return The shape of the buffer
 */
std::vector<int64_t> GetBufferShape(const Buffer& buffer, arith::Analyzer* analyzer) {
  int ndim = buffer->shape.size();
  std::vector<int64_t> result;
  result.reserve(ndim);
  for (const PrimExpr& i : buffer->shape) {
    if (const IntImmNode* int_imm = i.as<IntImmNode>()) {
      result.push_back(int_imm->value);
      continue;
    }
    arith::ConstIntBound bound = analyzer->const_int_bound(i);
    if (0 <= bound->max_value && bound->max_value < arith::ConstIntBound::kPosInf) {
      result.push_back(bound->max_value);
    } else {
      result.push_back(1);
    }
  }
  return result;
}

/*!
 * \brief Given a loop, return its `pragma_auto_unroll_max_step` annotation if it exists
 * \param loop The loop to be checked
 * \return The value of `pragma_auto_unroll_max_step` if it exists, or -1 if it does not exist
 */
int64_t GetPragmaAutoUnroll(const ForNode* loop) {
  if (Optional<IntImm> auto_unroll = GetAnn<IntImm>(loop, tir::attr::pragma_auto_unroll_max_step)) {
    return auto_unroll.value()->value;
  }
  return -1;
}

/*!
 * \brief Given a list of loops, return the extent of the first loop if the list is not empty,
 * and the first loop has constant extent. Otherwise returns the default value given
 * \param loops The list of loops to be checked
 * \param default_value The default value to be returned if the list is empty or the first loop
 * does not have constant extent
 * \return The extent of the first loop if the list is not empty, or the first loop has constant
 * extent. Otherwise returns the default value
 */
int64_t FirstLoopExtent(const ForVec& loops, int64_t default_value) {
  if (!loops.empty()) {
    if (const int64_t* extent = GetLoopIntExtent(loops[0])) {
      return *extent;
    }
  }
  return default_value;
}

/*!
 * \brief Relax each of the multi-indexing pattern according to the domains bound in the analyzer,
 * and then union them into a single region
 * \param multi_index_pattern A list of multi-index pattern to be relaxed
 * \param numel The size of the single region after union
 * \param analyzer The analyzer that contains the domain information
 * \return The relaxed and unioned region
 */
IntVec RelaxAndUnion(const std::vector<MultiIndex>& multi_indices, int64_t* numel,
                     arith::Analyzer* analyzer) {
  *numel = 1;
  if (multi_indices.empty()) {
    return {};
  }
  int n_indices = multi_indices.size();
  int ndim = multi_indices[0].size();
  IntVec access_shape(ndim, 0);
  for (int i = 0; i < ndim; ++i) {
    int64_t minimum = arith::ConstIntBound::kPosInf;
    int64_t maximum = arith::ConstIntBound::kNegInf;
    for (int j = 0; j < n_indices; ++j) {
      arith::ConstIntBound bound = analyzer->const_int_bound(multi_indices[j][i]);
      minimum = std::min(minimum, bound->min_value);
      maximum = std::max(maximum, bound->max_value);
    }
    *numel *= maximum - minimum + 1;
    access_shape[i] = maximum - minimum + 1;
  }
  return access_shape;
}

/*!
 * \brief Given a list of multi-index pattern, return the minimal stride of a variable on it
 * \param multi_indices The list of multi-index pattern
 * \param buffer_stride The stride of the buffer
 * \param var The variable to be checked
 * \return The minimal stride of the variable on the multi-index pattern
 */
int64_t GetVarStride(const std::vector<MultiIndex>& multi_indices, const IntVec& buffer_stride,
                     const Var& var) {
  class CoefficientExtractor : private ExprVisitor {
   public:
    static int64_t Extract(const PrimExpr& expr, const Var& var) {
      CoefficientExtractor extractor(var);
      extractor.VisitExpr(expr);
      return (extractor.visited_var && !extractor.visited_mul && !extractor.visited_add)
                 ? 1
                 : (extractor.visited_var ? extractor.stride : 0);
    }

   private:
    explicit CoefficientExtractor(const Var& var)
        : var(var), stride(0), visited_var(false), visited_add(false), visited_mul(false) {}

    void VisitExpr_(const MulNode* node) override {
      ExprVisitor::VisitExpr_(node);
      if (visited_var && !visited_add) {
        if (const auto* a = node->a.as<IntImmNode>()) {
          visited_mul = true;
          stride = a->value;
        } else if (const auto* b = node->b.as<IntImmNode>()) {
          visited_mul = true;
          stride = b->value;
        }
      }
    }

    void VisitExpr_(const AddNode* node) override {
      ExprVisitor::VisitExpr_(node);
      if (visited_var && !visited_mul) {
        visited_add = true;
        stride = 1;
      }
    }

    void VisitExpr_(const VarNode* node) override {
      if (node == var.get()) {
        visited_var = true;
        stride = 2;
      }
    }

    const Var& var;
    int64_t stride;
    bool visited_var;
    bool visited_add;
    bool visited_mul;
  };

  constexpr int64_t kNotFound = std::numeric_limits<int64_t>::max();
  int ndim = buffer_stride.size();
  // Calculate the min stride possible
  int64_t result = kNotFound;
  for (const MultiIndex& multi_index : multi_indices) {
    ICHECK_EQ(multi_index.size(), buffer_stride.size());
    // Find the rightest dimension that contains the given variable
    for (int i = ndim - 1; i >= 0; --i) {
      int64_t coef = CoefficientExtractor::Extract(multi_index[i], var);
      if (coef != 0) {
        result = std::min(result, std::abs(coef) * buffer_stride[i]);
        break;
      }
    }
  }
  return (result == kNotFound) ? 0 : result;
}

/*!
 * \brief Converts a 2-dimensional STL vector to a TVM NDArray
 * \param src The source 2-dimensional STL vector
 * \param second_dim_size The length of the second dimension. When the first dim of src is 0,
 * second_dim_size must be specified, and in such case the shape of the result NDArray is
 * (0, second_dim_size).
 * \return The converted TVM NDArray
 */
runtime::NDArray AsNDArray(const std::vector<std::vector<double>>& src, int second_dim_size = -1) {
  int n = src.size();
  ICHECK(!src.empty() || second_dim_size != -1);
  int m = src.empty() ? second_dim_size : src[0].size();
  runtime::NDArray tgt = runtime::NDArray::Empty(
      /*shape=*/{n, m},
      /*dtype=*/DLDataType{kDLFloat, 64, 1},
      /*ctx=*/DLDevice{kDLCPU, 0});
  double* data = static_cast<double*>(tgt->data);
  for (const std::vector<double>& row : src) {
    for (double v : row) {
      *data++ = v;
    }
  }
  return tgt;
}

}  // namespace TN
}  // namespace utils


namespace transform {
namespace TN    {
/*!
 * \brief Create a pass that simplifies the IR for feature extraction
 * \return The pass created
 */
Pass SimplifyForFeatureExtraction() {
  class Simplifier : private StmtExprMutator {
   public:
    static Stmt Run(Stmt stmt) { return Simplifier()(std::move(stmt)); }

   private:
    static bool HasBufferLoad(const PrimExpr& expr) {
      bool found = false;
      PostOrderVisit(expr, [&found](const ObjectRef& node) {
        if (node->IsInstance<BufferLoadNode>()) {
          found = true;
        }
      });
      return found;
    }

    PrimExpr VisitExpr_(const SelectNode* node) final {
      if (HasBufferLoad(node->true_value) || HasBufferLoad(node->false_value) ||
          HasBufferLoad(node->condition)) {
        return GetRef<Select>(node);
      }
      return make_const(node->dtype, 1.0);
    }

    PrimExpr VisitExpr_(const VarNode* var) final {
      if (unit_vars_.count(GetRef<Var>(var))) {
        return make_const(var->dtype, 0.0);
      }
      return GetRef<Var>(var);
    }

    Stmt VisitStmt_(const ForNode* loop) final {
      if (is_zero(loop->min) && is_one(loop->extent) && loop->kind == ForKind::kSerial &&
          loop->annotations.empty()) {
        unit_vars_.insert(loop->loop_var);
        return VisitStmt(loop->body);
      } else {
        return StmtExprMutator::VisitStmt_(loop);
      }
    }

    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> unit_vars_;
  };
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    PrimFuncNode* n = f.CopyOnWrite();
    n->body = Simplifier::Run(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SimplifyForFeatureExtraction", {});
}

/*!
 * \brief Create a list of passes that preprocesses the IR for feature extraction
 * \return The list of passes created
 */
Sequential PassListForPerBlockFeature() {
  return Sequential({
      tir::transform::RemoveWeightLayoutRewriteBlock(/*skip_ndarray_rewrite*/ true),
    //   tir::transform::SimplifyForFeatureExtraction(),
      tir::transform::LowerCrossThreadReduction(),
      tir::transform::LowerInitBlock(),
      tir::transform::PlanAndUpdateBufferAllocationLocation(),
      tir::transform::ConvertBlocksToOpaque(),
      tir::transform::CompactBufferAllocation(),
      tir::transform::Simplify(),
      tir::transform::LowerAutoCopy(),
      // tir::transform::UnifyThreadBinding(),
      tir::transform::LowerMatchBuffer(),
      tir::transform::Simplify(),
  });
}
}  // namespace TN
}  // namespace transform

namespace TN    {
/*! \brief A data structure managing loop nests */
struct LoopNest {
  int64_t prod = 1;    // The product of the extents of all the loops
  ForVec loops;        // All the loops
  IntVec auto_unroll;  // The loops with auto unroll pragma
  ForVec parallel;     // The loops whose ForKind are kParallel
  ForVec vectorize;    // The loops whose ForKind are kVectorized
  ForVec unroll;       // The loops whose ForKind are kUnrolled
  ForVec blockIdx_x;   // The loops whose ForKind are kThreadBinding to blockIdx.x
  ForVec blockIdx_y;   // The loops whose ForKind are kThreadBinding to blockIdx.y
  ForVec blockIdx_z;   // The loops whose ForKind are kThreadBinding to blockIdx.z
  ForVec threadIdx_x;  // The loops whose ForKind are kThreadBinding to threadIdx.x
  ForVec threadIdx_y;  // The loops whose ForKind are kThreadBinding to threadIdx.y
  ForVec threadIdx_z;  // The loops whose ForKind are kThreadBinding to threadIdx.z
  ForVec vthread;      // The loops whose ForKind are kThreadBinding to vthread.*

  /*!
   * \brief Push a new loop into the loop nest
   * \param loop The loop to be pushed
   * \param auto_unroll_attr The auto unroll attribute of the loop
   * \return A list of for loops that the loop is bound to
   */
  ForVec* Push(const ForNode* loop, int64_t* auto_unroll_attr) {
    if (const int64_t* extent = GetLoopIntExtent(loop)) {
      this->prod *= *extent;
    }
    this->loops.push_back(loop);
    if ((*auto_unroll_attr = utils::TN::GetPragmaAutoUnroll(loop)) > 0) {
      this->auto_unroll.push_back(*auto_unroll_attr);
    }
    ForVec* ref_loops = nullptr;
    if (loop->kind == ForKind::kParallel) {
      ref_loops = &parallel;
    } else if (loop->kind == ForKind::kVectorized) {
      ref_loops = &vectorize;
    } else if (loop->kind == ForKind::kUnrolled) {
      ref_loops = &unroll;
    } else if (loop->kind == ForKind::kThreadBinding) {
      std::string thread_tag = loop->thread_binding.value()->thread_tag;
      if (thread_tag == "blockIdx.x") {
        ref_loops = &blockIdx_x;
      } else if (thread_tag == "blockIdx.y") {
        ref_loops = &blockIdx_y;
      } else if (thread_tag == "blockIdx.z") {
        ref_loops = &blockIdx_z;
      } else if (thread_tag == "threadIdx.x") {
        ref_loops = &threadIdx_x;
      } else if (thread_tag == "threadIdx.y") {
        ref_loops = &threadIdx_y;
      } else if (thread_tag == "threadIdx.z") {
        ref_loops = &threadIdx_z;
      } else if (support::StartsWith(thread_tag, "vthread")) {
        ref_loops = &vthread;
      } else {
        LOG(FATAL) << "ValueError: Unable to recognize thread tag: " << thread_tag;
      }
    }
    if (ref_loops != nullptr) {
      ref_loops->push_back(loop);
    }
    return ref_loops;
  }

  /*!
   * \brief Pop the last loop from the loop nest
   * \param loop The loop to be popped
   * \param ref_loops The list of for loops that the loop is bound to
   * \param auto_unroll_attr The auto unroll attribute of the loop
   */
  void Pop(const ForNode* loop, ForVec* ref_loops, int auto_unroll_attr) {
    if (ref_loops) {
      ref_loops->pop_back();
    }
    if (auto_unroll_attr > 0) {
      this->auto_unroll.pop_back();
    }
    if (const int64_t* extent = GetLoopIntExtent(loop)) {
      this->prod /= *extent;
    }
    this->loops.pop_back();
  }
};


/****** Group 1: Computation related features ******/

namespace group1 {

/*! \brief Group 1 features */
struct Feature {
  /*! \brief Arithmetic features */
  struct ArithOps {
    // Float-point arithmetic features
    int64_t float_mad = 0;         // The number of float MAD (Multiply–add) ops
    int64_t float_add_sub = 0;     // The number of float add and sub ops
    int64_t float_mul = 0;         // The number of float multiply ops
    int64_t float_div_mod = 0;     // The number of float div and mod ops
    int64_t float_cmp = 0;         // The number of float comparison ops
    int64_t float_math_func = 0;   // The number of float math func calls
    int64_t float_other_func = 0;  // The number of other float func calls
    // Integer arithmetic features
    int64_t int_mad = 0;         // The number of integer MAD (Multiply–add) ops
    int64_t int_add_sub = 0;     // The number of integer add and sub ops
    int64_t int_mul = 0;         // The number of integer multiply ops
    int64_t int_div_mod = 0;     // The number of integer div and mod ops
    int64_t int_cmp = 0;         // The number of integer comparison ops
    int64_t int_math_func = 0;   // The number of integer math func calls
    int64_t int_other_func = 0;  // The number of other integer func calls
    // Other arithmetic features
    int64_t bool_op = 0;    // The number of bool ops
    int64_t select_op = 0;  // The number of select ops

    static constexpr int64_t kCount = 16;

    ArithOps() = default;
    ArithOps(const BufferStoreNode* store, int64_t prod_loop_extent);

    void Export(std::vector<double>* v) const {
      double vs[] = {
          slog(float_mad), slog(float_add_sub),   slog(float_mul),        slog(float_div_mod),
          slog(float_cmp), slog(float_math_func), slog(float_other_func),  //
          slog(int_mad),   slog(int_add_sub),     slog(int_mul),          slog(int_div_mod),
          slog(int_cmp),   slog(int_math_func),   slog(int_other_func),  //
          slog(bool_op),   slog(select_op),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }
  };

  /*! \brief Loop binding features */
  struct ForKindFeature {
    enum class Pos : int {
      kPosNone = 0,           // Does not have this kind of annotation
      kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
      kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
      kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
      kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
      kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
      kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
      kPosMixed = 7,          // The annotated iterator is a mixed space and reduce iterator
    };
    int64_t num = 0;           // The number of iterators with the annotation
    int64_t prod = 0;          // The product of the lengths of iterators with the annotation
    int64_t len = 0;           // The length of the innermost iterator with the annotation
    Pos pos = Pos::kPosMixed;  // The position of the iterators with the annotation

    static constexpr int64_t kCount = 11;

    explicit ForKindFeature(const ForVec& loops);

    void Export(std::vector<double>* v) const {
      double vs[] = {
          slog(num),
          slog(prod),
          slog(len),
          static_cast<double>(static_cast<int>(pos) == 0),
          static_cast<double>(static_cast<int>(pos) == 1),
          static_cast<double>(static_cast<int>(pos) == 2),
          static_cast<double>(static_cast<int>(pos) == 3),
          static_cast<double>(static_cast<int>(pos) == 4),
          static_cast<double>(static_cast<int>(pos) == 5),
          static_cast<double>(static_cast<int>(pos) == 6),
          static_cast<double>(static_cast<int>(pos) == 7),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }
  };

  ArithOps arith_ops;           // Arithmetic features
  ForKindFeature vectorize;     // Loop binding features: kVectorize
  ForKindFeature unroll;        // Loop binding features: kUnroll
  ForKindFeature parallel;      // Loop binding features: kParallel
  bool is_gpu = false;          // If the program is running on GPU
  int64_t blockIdx_x_len = 1;   // The length of blockIdx.x
  int64_t blockIdx_y_len = 1;   // The length of blockIdx.y
  int64_t blockIdx_z_len = 1;   // The length of blockIdx.z
  int64_t threadIdx_x_len = 1;  // The length of threadIdx.x
  int64_t threadIdx_y_len = 1;  // The length of threadIdx.y
  int64_t threadIdx_z_len = 1;  // The length of threadIdx.z
  int64_t vthread_len = 1;      // The length of virtual thread

  static constexpr int64_t kCount = ArithOps::kCount + ForKindFeature::kCount * 3 + 8;

  explicit Feature(const BufferStoreNode* store, const LoopNest& loop_nest, bool is_gpu)
      : arith_ops(store, loop_nest.prod),
        vectorize(loop_nest.vectorize),
        unroll(loop_nest.unroll),
        parallel(loop_nest.parallel) {
    if (is_gpu) {
      this->is_gpu = true;
      this->blockIdx_x_len = utils::TN::FirstLoopExtent(loop_nest.blockIdx_x, 1);
      this->blockIdx_y_len = utils::TN::FirstLoopExtent(loop_nest.blockIdx_y, 1);
      this->blockIdx_z_len = utils::TN::FirstLoopExtent(loop_nest.blockIdx_z, 1);
      this->threadIdx_x_len = utils::TN::FirstLoopExtent(loop_nest.threadIdx_x, 1);
      this->threadIdx_y_len = utils::TN::FirstLoopExtent(loop_nest.threadIdx_y, 1);
      this->threadIdx_z_len = utils::TN::FirstLoopExtent(loop_nest.threadIdx_z, 1);
      this->vthread_len = utils::TN::FirstLoopExtent(loop_nest.vthread, 1);
    }
  }

  void Export(std::vector<double>* v) const {
    this->arith_ops.Export(v);
    this->vectorize.Export(v);
    this->unroll.Export(v);
    this->parallel.Export(v);
    double vs[] = {
        static_cast<double>(is_gpu),  //
        slog(blockIdx_x_len),        slog(blockIdx_y_len),  slog(blockIdx_z_len),
        slog(threadIdx_x_len),       slog(threadIdx_y_len), slog(threadIdx_z_len),
        slog(vthread_len),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }
};

Feature::ArithOps::ArithOps(const BufferStoreNode* store, int64_t prod_loop_extent) {
  class ArithOpCounter : public ExprVisitor {
   public:
#define TVM_FEATURE_SIMPLE(Type, Counter)       \
  void VisitExpr_(const Type* op) final {       \
    result_.Counter += this->prod_loop_extent_; \
    ExprVisitor::VisitExpr_(op);                \
  }
#define TVM_FEATURE_BINARY(Type, FloatCounter, IntCounter) \
  void VisitExpr_(const Type* op) final {                  \
    if (op->dtype.is_float()) {                            \
      result_.FloatCounter += this->prod_loop_extent_;     \
    } else {                                               \
      result_.IntCounter += this->prod_loop_extent_;       \
    }                                                      \
    ExprVisitor::VisitExpr_(op);                           \
  }
    TVM_FEATURE_SIMPLE(AndNode, bool_op);
    TVM_FEATURE_SIMPLE(OrNode, bool_op);
    TVM_FEATURE_SIMPLE(NotNode, bool_op);
    TVM_FEATURE_SIMPLE(SelectNode, select_op);
    TVM_FEATURE_BINARY(AddNode, float_add_sub, int_add_sub);
    TVM_FEATURE_BINARY(SubNode, float_add_sub, int_add_sub);
    TVM_FEATURE_BINARY(MulNode, float_mul, int_mul);
    TVM_FEATURE_BINARY(DivNode, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(ModNode, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(FloorDivNode, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(FloorModNode, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(MaxNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(MinNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(EQNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(NENode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(LTNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(LENode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(GTNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(GENode, float_cmp, int_cmp);
#undef TVM_FEATURE_BINARY
#undef TVM_FEATURE_SIMPLE

    void VisitExpr_(const CallNode* op) final {
      static auto op_call_effect_ = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
      TCallEffectKind effect_kind = op_call_effect_[Downcast<Op>(op->op)];
      bool is_pure =
          effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation;
      if (is_pure) {
        if (op->dtype.is_float()) {
          result_.float_math_func += prod_loop_extent_;
        } else {
          result_.int_math_func += prod_loop_extent_;
        }
      } else {
        if (op->dtype.is_float()) {
          result_.float_other_func += prod_loop_extent_;
        } else {
          result_.int_other_func += prod_loop_extent_;
        }
      }
      ExprVisitor::VisitExpr_(op);
    }

    int64_t prod_loop_extent_;
    ArithOps result_;
  };
  ArithOpCounter counter;
  counter.prod_loop_extent_ = prod_loop_extent;
  counter(store->value);
  *this = counter.result_;
}

Feature::ForKindFeature::ForKindFeature(const ForVec& loops) {
  if (loops.empty()) {
    this->num = 0;
    this->prod = 0;
    this->len = 0;
    this->pos = ForKindFeature::Pos::kPosNone;
  } else {
    const int64_t* last_loop_extent = GetLoopIntExtent(loops.back());
    this->num = loops.size();
    this->len = last_loop_extent ? *last_loop_extent : 1;
    this->pos = ForKindFeature::Pos::kPosMixed;
    int64_t& prod = this->prod = 1;
    for (const ForNode* loop : loops) {
      if (const int64_t* extent = GetLoopIntExtent(loop)) {
        prod *= *extent;
      }
    }
  }
}

}  // namespace group1

namespace group2 {

/*! \brief Group 2 features */
struct Feature {
  enum class AccessType : int {
    /*! The buffer is read but not written */
    kRead = 0,
    /*! The buffer is written but not read */
    kWrite = 1,
    /*! The buffer is both read and written */
    kReadWrite = 2,
    /*! Unknown type */
    kUnknownRW = 3,
  };
  enum class ReuseType : int {
    /*! Buffer reuse because accessed on each iteration of a loop */
    kLoopMultipleRead = 0,
    /*! Buffer reuse because it is serially accessed */
    kSerialMultipleReadWrite = 1,
    /*! No buffer reuse */
    kNoReuse = 2,
  };

  struct SubFeature {
    /*! \brief The buffer this feature is for */
    const BufferNode* buffer = nullptr;
    /*! \brief The access type of the buffer */
    AccessType access_type = AccessType::kUnknownRW;
    /*! \brief A list of multi-dimensonal indices used to access the buffer */
    std::vector<MultiIndex> multi_indices = {};
    // Access information
    /*! \brief loop_accessed_numel[i][...] means the number of elements accessed by loops[i] */
    std::vector<std::unordered_map<const BufferNode*, int64_t>> loop_accessed_numel = {};
    /*! \brief The shape of the data access */
    IntVec access_shape;
    /*! \brief The bytes that are continuously accessed */
    int64_t num_continuous_bytes = 1;
    // Stride information
    /*! \brief The min stride of the access */
    int64_t min_stride = 0;
    /*! \brief The innermost stride */
    int64_t innermost_stride = 0;
    /*! \brief The product of the non-strided loops */
    int64_t prod_non_strided_loop_extent = 0;
    // Reuse information
    /*! The type of data reuse */
    ReuseType reuse_type = ReuseType::kNoReuse;
    /*! The reuse distance in terms of number of iterations */
    double reuse_dis_iter = 0.0;
    /*! The reuse distance in terms of bytes */
    double reuse_dis_bytes = 0.0;
    /*! The reuse count */
    int64_t reuse_ct = 0;
    // Features
    /*! The touched memory in bytes */
    double bytes;
    /*! The touched unique memory in bytes */
    double unique_bytes;
    /*! The number of touched cache lines */
    double lines;
    /*! The number touched unique cache lines */
    double unique_lines;
    /*! bytes / reuse_ct */
    double bytes_d_reuse_ct;
    /*! unique_bytes / reuse_ct */
    double unique_bytes_d_reuse_ct;
    /*! lines / reuse_ct */
    double lines_d_reuse_ct;
    /*! unique_lines / reuse_ct */
    double unique_lines_d_reuse_ct;
    /*! The stride in access */
    double stride;

    static constexpr int64_t kCount = 18;

    void Export(std::vector<double>* v) const {
      double vs[] = {
          static_cast<double>(static_cast<int>(access_type) == 0),
          static_cast<double>(static_cast<int>(access_type) == 1),
          static_cast<double>(static_cast<int>(access_type) == 2),
          // FeatureSet::BufferAccess::AccessType::kUnknownRW is ignored
          slog(bytes),
          slog(unique_bytes),
          slog(lines),
          slog(unique_lines),
          static_cast<double>(static_cast<int>(reuse_type) == 0),
          static_cast<double>(static_cast<int>(reuse_type) == 1),
          static_cast<double>(static_cast<int>(reuse_type) == 2),
          slog(reuse_dis_iter),
          slog(reuse_dis_bytes),
          slog(reuse_ct),
          slog(bytes_d_reuse_ct),
          slog(unique_bytes_d_reuse_ct),
          slog(lines_d_reuse_ct),
          slog(unique_lines_d_reuse_ct),
          slog(stride),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }

    static void Pad(std::vector<double>* v) { v->insert(v->end(), 18, 0.0); }

    void SetStride(const LoopNest& loop_nest, arith::Analyzer* analyzer);

    void SetReuse(const LoopNest& loop_nest,     //
                  int64_t top_loop_touch_bytes,  //
                  const ForBufferMap<IntVec>& buffer_touched_under_loop);

    void SetFeature(const LoopNest& loop_nest, int64_t cache_line_bytes);

    explicit SubFeature(const BufferNode* buffer, AccessType access_type,
                        std::vector<MultiIndex> multi_indices, int n_loops)
        : buffer(buffer),
          access_type(access_type),
          multi_indices(multi_indices),
          loop_accessed_numel(n_loops) {}
  };

  void Export(std::vector<double>* v, int buffers_per_store) const {
    int n = sub_features.size();
    for (int i = 0; i < buffers_per_store; ++i) {
      if (i < n) {
        sub_features[i].Export(v);
      } else {
        SubFeature::Pad(v);
      }
    }
  }

  explicit Feature(const BufferStoreNode* store, const LoopNest& loop_nest,
                   int64_t cache_line_bytes, IntVec* for_touched_bytes,
                   ForBufferMap<IntVec>* buffer_touched_under_loop, arith::Analyzer* analyzer);

  void Init(const BufferStoreNode* store, int n_loops);

  void SetRegion(const LoopNest& loop_nest,                        //
                 IntVec* for_touched_bytes,                        //
                 ForBufferMap<IntVec>* buffer_touched_under_loop,  //
                 arith::Analyzer* analyzer);

  std::vector<SubFeature> sub_features;
};

void Feature::Init(const BufferStoreNode* store, int n_loops) {
  struct Info {
    AccessType access_type = AccessType::kUnknownRW;
    std::vector<MultiIndex> multi_indices;
  };
  std::unordered_map<const BufferNode*, Info> buffer_info;
  {
    Info& info = buffer_info[store->buffer.get()];
    info.access_type = AccessType::kWrite;
    info.multi_indices.push_back({store->indices.begin(), store->indices.end()});
  }
  PostOrderVisit(store->value, [&buffer_info](const ObjectRef& obj) -> void {
    if (const BufferLoadNode* load = obj.as<BufferLoadNode>()) {
      const BufferNode* buffer = load->buffer.get();
      Info& info = buffer_info[buffer];
      switch (info.access_type) {
        case AccessType::kRead:
          break;
        case AccessType::kWrite:
          info.access_type = AccessType::kReadWrite;
          break;
        case AccessType::kReadWrite:
          break;
        case AccessType::kUnknownRW:
        default:
          info.access_type = AccessType::kRead;
          break;
      }
      if (info.access_type != AccessType::kReadWrite) {
        info.multi_indices.push_back({load->indices.begin(), load->indices.end()});
      }
    }
  });
  this->sub_features.reserve(buffer_info.size());
  for (const auto& kv : buffer_info) {
    this->sub_features.emplace_back(kv.first, kv.second.access_type,
                                    std::move(kv.second.multi_indices), n_loops);
  }
}

void Feature::SetRegion(const LoopNest& loop_nest, IntVec* for_touched_bytes,
                        ForBufferMap<IntVec>* buffer_touched_under_loop,
                        arith::Analyzer* analyzer) {
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForNode*>& loops = loop_nest.loops;
  // Step 1. Initialize and bind all the loop variables to a constant
  *for_touched_bytes = IntVec(n_loops, 0);
  for (int i = 0; i < n_loops; ++i) {
    const ForNode* loop = loops[i];
    analyzer->Bind(loop->loop_var, loop->min, /*allow_override=*/true);
  }
  // Step 2. Corner case: no loops
  if (n_loops == 0) {
    // In this case, the `access_shape` is not calculated
    for (SubFeature& feature : sub_features) {
      feature.access_shape = IntVec(feature.buffer->shape.size(), 1);
    }
    return;
  }
  // Step 3. Gradually bind the loops from inner to outer,
  // calculate the area the loops touch on each buffer
  for (int i = n_loops - 1; i >= 0; --i) {
    const ForNode* loop = loops[i];
    analyzer->Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent),
                   /*allow_override=*/true);
    int64_t& touched_bytes = (*for_touched_bytes)[i] = 0;
    for (SubFeature& feature : sub_features) {
      const BufferNode* buffer = feature.buffer;
      // Note: `feature.access_shape` for `i == 0` is the only one preserved,
      // while others are discarded
      int64_t numel;
      feature.access_shape = utils::TN::RelaxAndUnion(feature.multi_indices, &numel, analyzer);
      numel = std::max<int64_t>(0, numel);
      feature.loop_accessed_numel[i][buffer] = numel;
      touched_bytes += numel * buffer->dtype.bytes();
      (*buffer_touched_under_loop)[loop][buffer].push_back(numel);
    }
  }
}

void Feature::SubFeature::SetStride(const LoopNest& loop_nest, arith::Analyzer* analyzer) {
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForNode*>& loops = loop_nest.loops;
  // For each buffer, we find the loop stride on it
  const BufferNode* buffer = this->buffer;
  int ndim = this->buffer->shape.size();
  IntVec buffer_shape = utils::TN::GetBufferShape(GetRef<Buffer>(buffer), analyzer);
  // Calculate the buffer's stride from its shape
  IntVec buffer_stride(ndim);
  if (ndim >= 1) {
    buffer_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
      buffer_stride[i] = buffer_stride[i + 1] * buffer_shape[i + 1];
    }
  }
  // Calculate `num_continuous_bytes`
  {
    int64_t& num_continuous_bytes = this->num_continuous_bytes = 1;
    const IntVec& access_shape = this->access_shape;
    ICHECK_EQ(access_shape.size(), buffer_shape.size());
    for (int i = ndim - 1; i >= 0; --i) {
      if (access_shape[i] == buffer_shape[i]) {
        num_continuous_bytes = buffer_shape[i] * buffer->dtype.bytes();
        break;
      }
    }
  }
  // Enumerate loops from inner to outer
  int i = 0;
  // Calculate this->min_stride
  int64_t& stride = this->min_stride = 0;
  for (i = n_loops - 1; i >= 0; --i) {
    stride = utils::TN::GetVarStride(this->multi_indices, buffer_stride, loops[i]->loop_var);
    if (stride != 0) {
      break;
    }
  }
  // Calculate this->innermost_stride
  this->innermost_stride = (i == n_loops - 1) ? stride : 0;
  // Calculate this->prod
  int64_t& prod = this->prod_non_strided_loop_extent = 1;
  for (int j = n_loops - 1; j > i; --j) {
    if (const int64_t* extent = GetLoopIntExtent(loops[j])) {
      prod *= *extent;
    }
  }
}

void Feature::SubFeature::SetReuse(const LoopNest& loop_nest, int64_t top_loop_touch_bytes,
                                   const ForBufferMap<IntVec>& buffer_touched_under_loop) {
  const BufferNode* buffer = this->buffer;
  // Step 3.1. Collect all `Var`s that appears in the buffer region
  std::unordered_set<const VarNode*> region_vars;
  for (const MultiIndex& multi_index : this->multi_indices) {
    for (const PrimExpr& index : multi_index) {
      PostOrderVisit(index, [&region_vars](const ObjectRef& obj) -> void {
        if (const auto* var = obj.as<VarNode>()) {
          region_vars.insert(var);
        }
      });
    }
  }
  // Default case: no reuse
  ReuseType& reuse_type = this->reuse_type = ReuseType::kNoReuse;
  double& reuse_dis_iter = this->reuse_dis_iter = 0;
  double& reuse_dis_bytes = this->reuse_dis_bytes = 0;
  int64_t& reuse_ct = this->reuse_ct = 0;

  // Step 3.2. Enumerate loops from inner to outer, find the first loop with reuse
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForNode*>& loops = loop_nest.loops;
  for (int i = n_loops - 1; i >= 0; --i) {
    const ForNode* loop = loops[i];
    // Case 1. Find an invariant loop, i.e. reuse with kLoopMultipleRead
    if (!region_vars.count(loop->loop_var.get())) {
      reuse_type = ReuseType::kLoopMultipleRead;
      if (const int64_t* extent = GetLoopIntExtent(loop)) {
        reuse_ct = *extent;
      } else {
        reuse_ct = 1;
      }
      reuse_dis_iter = 1;
      for (int j = n_loops - 1; j > i; --j) {
        if (const int64_t* extent = GetLoopIntExtent(loops[j])) {
          reuse_dis_iter *= *extent;
        }
      }
      reuse_dis_bytes = 0.0;
      if (i == n_loops - 1) {
        reuse_dis_bytes = top_loop_touch_bytes;
      } else {
        for (const auto& iter : buffer_touched_under_loop.at(loops[i + 1])) {
          const BufferNode* buffer = iter.first;
          const IntVec& numels = iter.second;
          int64_t numel = std::accumulate(numels.begin(), numels.end(), int64_t(0));
          reuse_dis_bytes += numel * buffer->dtype.bytes();
        }
      }
      break;
    }
    // Case 2. Find serial reuse, i.e. reuse with kSerialMultipleReadWrite
    const IntVec& touched = buffer_touched_under_loop.at(loop).at(buffer);
    if (touched.size() >= 2) {
      int64_t extent = 1;
      if (const int64_t* ext = GetLoopIntExtent(loop)) {
        extent = *ext;
      }
      reuse_type = ReuseType::kSerialMultipleReadWrite;
      reuse_ct = touched.size() - 1;
      reuse_dis_iter = *std::min_element(touched.begin(), touched.end());
      reuse_dis_bytes = 0.0;
      for (const auto& iter : buffer_touched_under_loop.at(loop)) {
        const BufferNode* buffer = iter.first;
        const IntVec& numels = iter.second;
        int64_t numel = std::accumulate(numels.begin(), numels.end(), int64_t(0));
        reuse_dis_bytes += numel * buffer->dtype.bytes();
      }
      reuse_dis_iter /= extent;
      reuse_dis_bytes /= extent;
      break;
    }
  }
}

void Feature::SubFeature::SetFeature(const LoopNest& loop_nest, int64_t cache_line_bytes) {
  int64_t dtype_bytes = this->buffer->dtype.bytes();
  this->stride = this->innermost_stride;
  this->bytes = dtype_bytes * loop_nest.prod;
  if (loop_nest.loops.empty()) {
    this->unique_bytes = 1;
    this->lines = 1;
    this->unique_lines = 1;
  } else {
    this->unique_bytes =
        static_cast<double>(this->loop_accessed_numel.front().at(buffer)) * dtype_bytes;
    this->lines = static_cast<double>(loop_nest.prod) / this->prod_non_strided_loop_extent *
                  std::min(1.0, 1.0 * this->min_stride * dtype_bytes / cache_line_bytes);
    this->lines = std::max(1.0, this->lines);
    this->unique_lines = static_cast<double>(this->unique_bytes) /
                         std::min(cache_line_bytes, this->num_continuous_bytes);
    this->unique_lines = std::max(1.0, this->unique_lines);
  }
  double proxy_reuse_ct = this->reuse_ct > 0 ? this->reuse_ct : 0.5;
  this->bytes_d_reuse_ct = this->bytes / proxy_reuse_ct;
  this->unique_bytes_d_reuse_ct = this->unique_bytes / proxy_reuse_ct;
  this->lines_d_reuse_ct = this->lines / proxy_reuse_ct;
  this->unique_lines_d_reuse_ct = this->unique_lines / proxy_reuse_ct;
}

Feature::Feature(const BufferStoreNode* store, const LoopNest& loop_nest, int64_t cache_line_bytes,
                 IntVec* for_touched_bytes, ForBufferMap<IntVec>* buffer_touched_under_loop,
                 arith::Analyzer* analyzer) {
  int n_loops = loop_nest.loops.size();
  // Step 0. Initialize data structures
  this->Init(store, n_loops);
  // Step 1. Calculate region-related feature
  this->SetRegion(loop_nest, for_touched_bytes, buffer_touched_under_loop, analyzer);
  // Step 2. Calculate stride-related feature
  for (auto& feature : sub_features) {
    feature.SetStride(loop_nest, analyzer);
  }
  // Step 3. Calculate reuse-related feature
  int64_t top_loop_touch_bytes = 0.0;
  if (n_loops > 0) {
    for (const SubFeature& feature : sub_features) {
      int64_t bytes = feature.buffer->dtype.bytes();
      int64_t n_buffer = feature.loop_accessed_numel[0].size();
      top_loop_touch_bytes += bytes * n_buffer;
    }
  }
  for (auto& feature : sub_features) {
    feature.SetReuse(loop_nest, top_loop_touch_bytes, *buffer_touched_under_loop);
  }
  // Step 4. Calculate rest of the features
  for (auto& feature : sub_features) {
    feature.SetFeature(loop_nest, cache_line_bytes);
  }
  // Step 5. Sort the features
  std::sort(sub_features.begin(), sub_features.end(), [](const SubFeature& a, const SubFeature& b) {
    if (a.lines != b.lines) {
      return a.lines > b.lines;
    }
    if (a.bytes != b.bytes) {
      return a.bytes > b.bytes;
    }
    return a.buffer->name < b.buffer->name;
  });
}

}  // namespace group2

namespace group3 {

/*! \brief Group 3 feature */
struct Feature {
  /*!
   * \brief See the wiki page [1] for details
   *
   * Arithmetic intensity is FLOPs/unique bytes of memory touched. A value is computed
   * for each set of loop nests starting with just the innermost loop and
   * reaching to include all loops. There are a variable number of loops, so
   * n_samples are taken from the curve of arithmetic intensity vs flops. This
   * biases the values towards larger loops.
   *
   * Note that the denominator is unique bytes of memory touched. Repeated
   * access to the same byte of memory counts as only a single byte touched.
   *
   * Values are scaled by log2(x + 1).
   *
   * [1] https://en.wikipedia.org/wiki/Roofline_model
   */
  std::vector<double> arith_intensity_curve;

  void Export(std::vector<double>* v) const {
    v->insert(v->end(), arith_intensity_curve.begin(), arith_intensity_curve.end());
  }

  explicit Feature(int n_samples, const LoopNest& loop_nest, const IntVec& for_touched_bytes,
                   const group1::Feature::ArithOps& arith_ops)
      : arith_intensity_curve(n_samples, 0.0) {
    const std::vector<const ForNode*>& loops = loop_nest.loops;
    ICHECK_EQ(loops.size(), for_touched_bytes.size());
    int n_loops = loops.size();
    // Calculate `memory_bytes`
    std::vector<double> memory_bytes;
    memory_bytes.resize(n_loops);
    for (int i = 0; i < n_loops; ++i) {
      memory_bytes[n_loops - 1 - i] = for_touched_bytes[i];
    }
    // Calculate `compute_ops` and `cur_compute_ops`
    std::vector<double> compute_ops;
    double total_compute_ops = arith_ops.float_mad + arith_ops.float_add_sub + arith_ops.float_mul +
                               arith_ops.float_div_mod + arith_ops.float_cmp +
                               arith_ops.float_math_func + arith_ops.float_other_func;
    total_compute_ops /= loop_nest.prod;
    for (int i = n_loops - 1; i >= 0; --i) {
      if (const int64_t* extent = GetLoopIntExtent(loops[i])) {
        total_compute_ops *= *extent;
      }
      compute_ops.push_back(total_compute_ops);
    }
    // Fill the feature set
    if (total_compute_ops <= 0 || compute_ops.empty()) {
      for (int i = 0; i < n_samples; ++i) {
        arith_intensity_curve[i] = 0.0;
      }
      return;
    }
    total_compute_ops = compute_ops.back();
    int p = 0;
    for (int i = 0; i < n_samples; ++i) {
      double& result = arith_intensity_curve[i];
      double cur_compute_ops = static_cast<double>(i + 1) / n_samples * total_compute_ops;
      // Find the first `p` that `compute[p] >= total * (i + 1) / N`
      for (; p < n_loops; ++p) {
        if (compute_ops[p] >= cur_compute_ops - 1e-4) {
          break;
        }
      }
      CHECK_LT(p, n_loops);
      if (p == 0) {
        result = slog(compute_ops[p] / memory_bytes[p]);
      } else {
        double base = compute_ops[p - 1] / memory_bytes[p - 1];
        double slope =
            (compute_ops[p] / memory_bytes[p] - compute_ops[p - 1] / memory_bytes[p - 1]) /
            (compute_ops[p] - compute_ops[p - 1]);
        result = slog(base + slope * (cur_compute_ops - compute_ops[p - 1]));
      }
    }
  }
};

}  // namespace group3

namespace group4 {

/*! \brief Group 4 feature */
struct Feature {
  int64_t alloc_size = 0;        // The size of allocated buffer in bytes
  int64_t alloc_prod = 0;        // alloc_outer_prod * alloc_inner_prod
  int64_t alloc_outer_prod = 1;  // The product of lengths of loops outside the scope of the alloc

  static constexpr int64_t kCount = 4;

  void Export(std::vector<double>* v, int64_t outer_prod) const {
    double vs[] = {
        slog(alloc_size),
        slog(alloc_prod),
        slog(alloc_outer_prod),
        slog(static_cast<double>(outer_prod) / alloc_outer_prod),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }

  Feature() = default;

  explicit Feature(const LoopNest& loop_nest, const Buffer& buffer, arith::Analyzer* analyzer) {
    std::vector<int64_t> shape = utils::TN::GetBufferShape(buffer, analyzer);
    int64_t numel = 1;
    for (int64_t x : shape) {
      numel *= x;
    }
    alloc_size = numel * buffer->dtype.bytes();
    alloc_prod = numel * loop_nest.prod;
    alloc_outer_prod = loop_nest.prod;
  }
};

}  // namespace group4

namespace group5 {

/*! \brief Group 5 feature */
struct Feature {
  int64_t outer_prod;        // The product of lengths of outer loops
  int num_loops;             // The number of outer loops
  int auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"

  static constexpr int64_t kCount = 3;

  void Export(std::vector<double>* v) const {
    double vs[] = {
        slog(outer_prod),
        slog(num_loops),
        slog(auto_unroll_max_step),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }

  explicit Feature(const LoopNest& loop_nest) {
    this->outer_prod = loop_nest.prod;
    this->num_loops = loop_nest.loops.size();
    this->auto_unroll_max_step = loop_nest.auto_unroll.empty() ? 0 : loop_nest.auto_unroll.back();
  }
};

}  // namespace group5

namespace group6 {

/*! \brief The auxiliary feature extractor for workloads */
class WorkloadEmbeddingExtractor : private StmtVisitor {
 public:
  static std::vector<double> Extract(const IRModule& mod) {
    WorkloadEmbeddingExtractor self;
    for (const auto& kv : mod->functions) {
      if (const PrimFuncNode* func = kv.second.as<PrimFuncNode>()) {
        self(func->body);
      }
    }
    return self.embedding;
  }

 private:
  void VisitStmt_(const BlockNode* block) final {
    StmtVisitor::VisitStmt_(block);
    std::string name = block->name_hint;
    std::for_each(name.begin(), name.end(), [](char& c) { c = ::tolower(c); });
    if (name.find("softmax") != std::string::npos) {
      embedding[0] = 1.0;
    } else if ((name.find("max") != std::string::npos) || (name.find("min") != std::string::npos)) {
      embedding[1] = 1.0;
    } else if (name.find("add") != std::string::npos) {
      embedding[2] = 1.0;
    } else if (name.find("batch_matmul") != std::string::npos) {
      embedding[3] = 1.0;
    } else if (name.find("matmul") != std::string::npos) {
      embedding[4] = 1.0;
    } else if (name.find("depthwiseconv2d") != std::string::npos) {
      embedding[5] = 1.0;
    } else if (name.find("conv2d_winograd") != std::string::npos) {
      embedding[6] = 1.0;
    } else if (name.find("conv2d") != std::string::npos) {
      embedding[7] = 1.0;
    }
  }

  std::vector<double> embedding = std::vector<double>(8, 0.0);
};

/*! \brief Group 6 feature */
struct Feature {
  explicit Feature(const IRModule& mod) {
    this->feature = WorkloadEmbeddingExtractor::Extract(mod);
  }

  void Export(std::vector<double>* v) const {
    v->insert(v->end(), std::begin(feature), std::end(feature));
  }

  std::vector<double> feature;  // The workload embedding
  static constexpr int64_t kCount = 8;
};

}  // namespace group6

/*! \brief The feature extracted */
struct Feature {
  const BufferNode* buffer = nullptr;
  int buffer_order = -1;
  std::shared_ptr<group1::Feature> group1 = nullptr;
  std::shared_ptr<group2::Feature> group2 = nullptr;
  std::shared_ptr<group3::Feature> group3 = nullptr;
  std::shared_ptr<group4::Feature> group4 = nullptr;
  std::shared_ptr<group5::Feature> group5 = nullptr;
  std::shared_ptr<group6::Feature> group6 = nullptr;

  bool operator<(const Feature& other) const { return buffer_order < other.buffer_order; }
};

/*********Tensorize-Notation*********/
class TNGlobalTable{
public:
    TNGlobalTable():
    GeneralPreprocess(0), IntrinPreprocess(0),
    GeneralLoad(0), IntrinLoad(0),GeneralCompute(0),IntrinCompute(0), 
    GeneralStore(0),IntrinStore(0), GeneralEpilogue(0), IntrinEpilogue(0),
    src2dest({0, 0}), blockcnt(0), curblockcnt(0)
    { 
      IntrinMap["wmma_sync_16x16x16_f16f16f16"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,16., 0., 0., 0., 2.,16., 0., 0., 0.,
          0.        ,12.00035218,12.00035218, 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 5.04439412, 4.08746284,
          4.08746284,13.0001761 , 9.00281502, 9.00281502, 9.00281502, 5.04439412,
          4.08746284, 4.08746284, 5.04439412, 1.        , 4.08746284,13.0001761 ,
          9.00281502, 9.00281502, 5.04439412, 5.04439412, 4.08746284, 4.08746284,
          5.04439412, 1.        , 4.08746284,13.0001761 , 9.00281502, 9.00281502,
          5.04439412, 1.5849625 , 4.08746284, 4.08746284
        });
      IntrinMap["wmma_sync_16x16x16_f16f16f16_trans"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,16., 0., 0., 0., 2.,16., 0., 0., 0.,
          0.        ,12.00035218,12.00035218, 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 5.04439412, 1.        ,
          4.08746284,13.0001761 , 9.00281502, 9.00281502, 5.04439412, 5.04439412,
          4.08746284, 4.08746284, 5.04439412, 1.        , 4.08746284,13.0001761 ,
          9.00281502, 9.00281502, 9.00281502, 5.04439412, 4.08746284, 4.08746284,
          5.04439412, 1.        , 4.08746284,13.0001761 , 9.00281502, 9.00281502,
          5.04439412, 1.5849625 , 4.08746284, 4.08746284
        });
      IntrinMap["wmma_load_16x16x16_f16_a_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,5.04439412,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,5.04439412,4.08746284,
          4.08746284,5.04439412,1.        ,0.        ,9.00281502,9.00281502,
          9.00281502,5.04439412,4.08746284,4.08746284
        });
      IntrinMap["wmma_load_16x16x16_f16_b_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,5.04439412,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,5.04439412,4.08746284,
          4.08746284,5.04439412,1.        ,0.        ,9.00281502,9.00281502,
          9.00281502,5.04439412,4.08746284,4.08746284
        });
      IntrinMap["wmma_load_16x16x16_f16_a_trans_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,5.04439412,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,5.04439412,4.08746284,
          4.08746284,5.04439412,1.        ,0.        ,9.00281502,9.00281502,
          9.00281502,5.04439412,4.08746284,4.08746284
        });
      IntrinMap["wmma_load_16x16x16_f16_b_trans_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,5.04439412,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,5.04439412,4.08746284,
          4.08746284,5.04439412,1.        ,0.        ,9.00281502,9.00281502,
          9.00281502,5.04439412,4.08746284,4.08746284
        });
      IntrinMap["wmma_fill_16x16x16_f16"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,5.04439412,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,5.04439412,4.08746284,
          4.08746284
        });
      IntrinMap["wmma_store_16x16x16_f16_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,5.04439412,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,5.04439412,4.08746284,
          4.08746284,5.04439412,1.        ,0.        ,9.00281502,9.00281502,
          9.00281502,5.04439412,4.08746284,4.08746284
        });
      

      IntrinMap["wmma_sync_32x8x16_f16f16f16"] = \
        std::vector<double>({
          0.,32., 0., 0., 0., 1., 8., 0., 0., 0., 2.,16., 0., 0., 0.,
          0.        ,12.00035218,12.00035218, 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 4.08746284, 3.169925  ,
          5.04439412,13.0001761 , 8.00562455, 8.00562455, 8.00562455, 5.04439412,
          4.08746284, 3.169925  , 5.04439412, 1.        , 3.169925  ,13.0001761 ,
         10.00140819,10.00140819, 5.04439412, 5.04439412, 5.04439412, 4.08746284,
          4.08746284, 1.        , 4.08746284,13.0001761 , 9.00281502, 9.00281502,
          4.08746284, 1.5849625 , 5.04439412, 3.169925
        });
      IntrinMap["wmma_sync_32x8x16_f16f16f16_trans"] = \
        std::vector<double>({
          0.,32., 0., 0., 0., 1., 8., 0., 0., 0., 2.,16., 0., 0., 0.,
          0.        ,12.00035218,12.00035218, 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 5.04439412, 1.        ,
          3.169925  ,13.0001761 ,10.00140819,10.00140819, 5.04439412, 5.04439412,
          5.04439412, 4.08746284, 5.04439412, 1.        , 5.04439412,13.0001761 ,
          8.00562455, 8.00562455, 8.00562455, 5.04439412, 3.169925  , 4.08746284,
          4.08746284, 1.        , 4.08746284,13.0001761 , 9.00281502, 9.00281502,
          4.08746284, 1.5849625 , 5.04439412, 3.169925
        });
      IntrinMap["wmma_load_32x8x16_f16_a_shared_dyn"] = \
        std::vector<double>({
          0.,32., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 5.04439412, 1.        ,
          0.        ,10.00140819,10.00140819,10.00140819, 5.04439412, 5.04439412,
          4.08746284, 5.04439412, 1.        , 0.        ,10.00140819,10.00140819,
         10.00140819, 5.04439412, 5.04439412, 4.08746284
        });
      IntrinMap["wmma_load_32x8x16_f16_b_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1., 8., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,4.08746284,1.        ,
          0.        ,8.00562455,8.00562455,8.00562455,4.08746284,4.08746284,
          3.169925  ,4.08746284,1.        ,0.        ,8.00562455,8.00562455,
          8.00562455,4.08746284,4.08746284,3.169925
        });
      IntrinMap["wmma_load_32x8x16_f16_a_trans_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,32., 0., 0., 0.,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 6.02236781, 1.        ,
          0.        ,10.00140819,10.00140819,10.00140819, 6.02236781, 4.08746284,
          5.04439412, 6.02236781, 1.        , 0.        ,10.00140819,10.00140819,
         10.00140819, 6.02236781, 4.08746284, 5.04439412
        });
      IntrinMap["wmma_load_32x8x16_f16_b_trans_shared_dyn"] = \
        std::vector<double>({
          0., 8., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,5.04439412,1.        ,
          0.        ,8.00562455,8.00562455,8.00562455,5.04439412,3.169925  ,
          4.08746284,5.04439412,1.        ,0.        ,8.00562455,8.00562455,
          8.00562455,5.04439412,3.169925  ,4.08746284
        });
      IntrinMap["wmma_fill_32x8x16_f16"] = \
        std::vector<double>({
          0.,32., 0., 0., 0., 1., 8., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,4.08746284,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,4.08746284,5.04439412,
          3.169925
        });
      IntrinMap["wmma_store_32x8x16_f16_shared_dyn"] = \
        std::vector<double>({
          0.,32., 0., 0., 0., 1., 8., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,4.08746284,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,4.08746284,5.04439412,
          3.169925  ,4.08746284,1.        ,0.        ,9.00281502,9.00281502,
          9.00281502,4.08746284,5.04439412,3.169925
        });


      IntrinMap["wmma_sync_8x32x16_f16f16f16"] = \
        std::vector<double>({
          0., 8., 0., 0., 0., 1.,32., 0., 0., 0., 2.,16., 0., 0., 0.,
          0.        ,12.00035218,12.00035218, 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 6.02236781, 5.04439412,
          3.169925  ,13.0001761 ,10.00140819,10.00140819,10.00140819, 5.04439412,
          4.08746284, 5.04439412, 5.04439412, 1.        , 5.04439412,13.0001761 ,
          8.00562455, 8.00562455, 5.04439412, 5.04439412, 3.169925  , 4.08746284,
          6.02236781, 1.        , 4.08746284,13.0001761 , 9.00281502, 9.00281502,
          6.02236781, 1.5849625 , 3.169925  , 5.04439412,
        });
      IntrinMap["wmma_sync_8x32x16_f16f16f16_trans"] = \
        std::vector<double>({
          0., 8., 0., 0., 0., 1.,32., 0., 0., 0., 2.,16., 0., 0., 0.,
          0.        ,12.00035218,12.00035218, 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 5.04439412, 1.        ,
          5.04439412,13.0001761 , 8.00562455, 8.00562455, 5.04439412, 5.04439412,
          3.169925  , 4.08746284, 5.04439412, 1.        , 3.169925  ,13.0001761 ,
         10.00140819,10.00140819,10.00140819, 5.04439412, 5.04439412, 4.08746284,
          6.02236781, 1.        , 4.08746284,13.0001761 , 9.00281502, 9.00281502,
          6.02236781, 1.5849625 , 3.169925  , 5.04439412
        });
      IntrinMap["wmma_load_8x32x16_f16_a_shared_dyn"] = \
        std::vector<double>({
          0., 8., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,5.04439412,1.        ,
          0.        ,8.00562455,8.00562455,8.00562455,5.04439412,3.169925  ,
          4.08746284,5.04439412,1.        ,0.        ,8.00562455,8.00562455,
          8.00562455,5.04439412,3.169925  ,4.08746284
        });
      IntrinMap["wmma_load_8x32x16_f16_b_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1.,32., 0., 0., 0.,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 6.02236781, 1.        ,
          0.        ,10.00140819,10.00140819,10.00140819, 6.02236781, 4.08746284,
          5.04439412, 6.02236781, 1.        , 0.        ,10.00140819,10.00140819,
         10.00140819, 6.02236781, 4.08746284, 5.04439412
        });
      IntrinMap["wmma_load_8x32x16_f16_a_trans_shared_dyn"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1., 8., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,4.08746284,1.        ,
          0.        ,8.00562455,8.00562455,8.00562455,4.08746284,4.08746284,
          3.169925  ,4.08746284,1.        ,0.        ,8.00562455,8.00562455,
          8.00562455,4.08746284,4.08746284,3.169925
        });
      IntrinMap["wmma_load_8x32x16_f16_b_trans_shared_dyn"] = \
        std::vector<double>({
          0.,32., 0., 0., 0., 1.,16., 0., 0., 0.,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
          0.        , 0.        , 0.        , 0.        , 5.04439412, 1.        ,
          0.        ,10.00140819,10.00140819,10.00140819, 5.04439412, 5.04439412,
          4.08746284, 5.04439412, 1.        , 0.        ,10.00140819,10.00140819,
         10.00140819, 5.04439412, 5.04439412, 4.08746284
        });
      IntrinMap["wmma_fill_8x32x16_f16"] = \
        std::vector<double>({
          0., 8., 0., 0., 0., 1.,32., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,6.02236781,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,6.02236781,3.169925  ,
          5.04439412
        });
      IntrinMap["wmma_store_8x32x16_f16_shared_dyn"] = \
        std::vector<double>({
          0., 8., 0., 0., 0., 1.,32., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,6.02236781,1.        ,
          0.        ,9.00281502,9.00281502,9.00281502,6.02236781,3.169925  ,
          5.04439412,6.02236781,1.        ,0.        ,9.00281502,9.00281502,
          9.00281502,6.02236781,3.169925  ,5.04439412
        });



      IntrinMap["dot_16x4_avx512"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1., 4., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,6.02236781,6.02236781,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,6.02236781,1.        ,
          2.32192809,8.00562455,6.02236781,6.02236781,2.32192809,4.08746284,
          2.32192809,1.        ,4.08746284,6.02236781,2.32192809,2.32192809,
          2.32192809,2.32192809,2.32192809,1.        ,0.        ,6.02236781,
          6.02236781,6.02236781,2.32192809,4.08746284,2.32192809
        });
      IntrinMap["dot_16x4_vnni"] = \
        std::vector<double>({
          0.,16., 0., 0., 0., 1., 4., 0., 0., 0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,6.02236781,6.02236781,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,6.02236781,1.        ,
          2.32192809,8.00562455,6.02236781,6.02236781,2.32192809,4.08746284,
          2.32192809,1.        ,4.08746284,6.02236781,2.32192809,2.32192809,
          2.32192809,2.32192809,2.32192809,1.        ,0.        ,6.02236781,
          6.02236781,6.02236781,2.32192809,4.08746284,2.32192809
        });
      IntrinMap["dot_4x4_i8i8s32_neon"] = \
        std::vector<double>({
          0.,4.,0.,0.,0.,1.,4.,0.,0.,0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,4.08746284,4.08746284,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,4.08746284,1.        ,
          2.32192809,6.02236781,4.08746284,4.08746284,2.32192809,2.32192809,
          2.32192809,1.        ,2.32192809,4.08746284,2.32192809,2.32192809,
          2.32192809,2.32192809,2.32192809,1.        ,0.        ,4.08746284,
          4.08746284,4.08746284,2.32192809,2.32192809,2.32192809
        });
      IntrinMap["dot_4x4_i8i8s32_sdot"] = \
        std::vector<double>({
          0.,4.,0.,0.,0.,1.,4.,0.,0.,0.,
          0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
          0.        ,0.        ,4.08746284,4.08746284,0.        ,0.        ,
          0.        ,0.        ,0.        ,0.        ,4.08746284,1.        ,
          2.32192809,6.02236781,4.08746284,4.08746284,2.32192809,2.32192809,
          2.32192809,1.        ,2.32192809,4.08746284,2.32192809,2.32192809,
          2.32192809,2.32192809,2.32192809,1.        ,0.        ,4.08746284,
          4.08746284,4.08746284,2.32192809,2.32192809,2.32192809
        });
      
    };
    int64_t loop_id_counter = 0;
    int64_t Loop2Loopid(const ForNode* loop){
        auto it = loop_map.find(loop);
        if(it != loop_map.end()){
            return it->second;
        }
        else{
            throw std::out_of_range("loop not found");
        }
    }

    void AddLoop(const ForNode* loop){
        loop_map.insert({loop, loop_id_counter});
        loop_var_map.insert({loop->loop_var.get(), loop_id_counter});
        loop_id_counter ++;
    }

    void DelLoop(const ForNode* loop){
        auto it = loop_map.find(loop);
        if(it != loop_map.end()){
            loop_map.erase(it);
        }
        else{
            throw std::out_of_range("del loop_name_hint not found");
        }
    }
    
    // std::string GetGeneralBehavior(){
    //   if(!IntrinCompute && !IntrinLoad){
    //     return "Load";
    //   }
    //   if(IntrinCompute && IntrinStore){
    //     return "Store";
    //   }
    //   else{
    //     throw std::out_of_range("error in GetGeneralBehavior ...");
    //   }
    // }

    std::string GetGeneralBehavior(){
      std::string behavior("");
      // src 2 dest
      if(IntrinPreprocess){
        // GPU
        if(!IntrinCompute){
          behavior =  "Load";
          GeneralLoad += 1;
        }
        if(IntrinCompute){
          behavior =  "Store";
          GeneralStore += 1;
        }
        if(behavior == "Load"){
          src2dest.first = 1; src2dest.second = 2;
        }
        else if(behavior == "Store"){
          src2dest.first = 2; src2dest.second = 1;
        }
      }
      else if(IntrinPreprocess == 0){
        // CPU
        if(!IntrinCompute){
          if(std::string(CurBlock_affinehelper->name_hint).find("init") != std::string::npos){
            behavior = "Preprocess";
            GeneralPreprocess += 1;
            src2dest.first = src2dest.second = 1; // global 2 global
          }
          else{
            behavior = "Load";
            GeneralLoad += 1;
            src2dest.first = src2dest.second = 1; // global 2 global
          }
        }
        if(IntrinCompute){
          if(std::string(CurBlock_affinehelper->name_hint).find("relu") != std::string::npos || 
            std::string(CurBlock_affinehelper->name_hint).find("add") != std::string::npos){
            behavior = "Store";
            GeneralEpilogue += 1;
            src2dest.first = src2dest.second = 1; // global 2 global
          }
          else{
            behavior = "Store";
            GeneralStore += 1;
            if(curblockcnt < blockcnt){
              // cache write
              src2dest.first = 3; src2dest.second = 2;  // reg 2 cache
            }
            else{
              if(GeneralStore == 1){
                src2dest.first = 3; src2dest.second = 1; // reg 2 global
              }
              else{
                src2dest.first = 2; src2dest.second = 1; // cache 2 global
              }
            }
          }
        }
      }
      else{
        throw std::out_of_range("error in GetGeneralBehavior IntrinPreprocess ...");
      }
      return behavior;
    }

    std::vector<double> GetIntrin(std::string Key){
      auto it = IntrinMap.find(Key);
        if(it != IntrinMap.end()){
            return it->second;
        }
        else{
            std::cout << Key << std::endl;
            throw std::out_of_range("intrin not found ...");
        }

    }

    std::string GetBufferVarDtype(std::string Key){
      auto it = buffer_data_dtype.find(Key);
      if(it != buffer_data_dtype.end()){
            return it->second;
        }
        else{
            std::cout << Key << std::endl;
            throw std::out_of_range("BufferVarDtype not found ...");
        }
    }

    std::unordered_map<const ForNode*, int64_t> loop_map;
    std::unordered_map<std::string, std::vector<double>> IntrinMap;
    std::unordered_map<std::string, std::string> buffer_data_dtype;


    // tensorcore wmma
    std::string tcname;
    std::string tctype_;
    std::string tcshape;
    std::string tcmatrixadtype; // f16 ...
    std::string tcmatrixalayout; // row_major, col_major
    std::string tcmatrixbdtype;
    std::string tcmatrixblayout;
    std::string tcmatrixcdtype;
    std::string tcmatrixclayout;
    std::string tccurmatrix;
    

    // BehaviorCNT
    int GeneralPreprocess;
    int IntrinPreprocess;
    int GeneralLoad;
    int IntrinLoad;
    int GeneralCompute;
    int IntrinCompute;
    int GeneralStore;
    int IntrinStore;
    int GeneralEpilogue;
    int IntrinEpilogue;

    // Affine
    std::unordered_map<const BlockNode*, std::vector<double>> AffineInBlock;
    const BlockNode* CurBlock_affinehelper;
    std::unordered_map<const VarNode*, int64_t> loop_var_map; // update add in AddLoop()
    std::unordered_map<std::string, int64_t> AffineOp{
      {"add", 1}, {"sub", 2}, {"mul", 3}, {"div", 4}, {"mod", 5}, {"floordiv", 6}, {"floormod", 7}
    };


    int64_t getAffineLoopVar(const VarNode* var){
      auto it = loop_var_map.find(var);
      if(it == loop_var_map.end()){
        throw std::out_of_range("getAffineLoopVar fail ...");
      }
      return it->second;
    }

    void addAffine(const BlockNode* block, std::vector<double>& affine_read_write){
      auto it = AffineInBlock.find(block);
      if(it != AffineInBlock.end()) {
        throw std::out_of_range("addAffine fail ...");
      }
      std::vector<double>& v = AffineInBlock[block];
      v.insert(v.end(), affine_read_write.begin(), affine_read_write.end());
    }

    std::vector<double> getAffine(){
      auto it = AffineInBlock.find(CurBlock_affinehelper);
      if(it == AffineInBlock.end()) {
        throw std::out_of_range("getAffine fail ...");
      }
      return it->second;
    }
/*************************************Affine*************************************/
    void MakeAffine(){
      const BlockNode* block = CurBlock_affinehelper;
      std::vector<double> affine_read_write; affine_read_write.reserve(156);

      class AffineCounter : public ExprVisitor {
        public:
        AffineCounter(TNGlobalTable& global_table_):global_table_(global_table_){x.reserve(150);}
        #define TVM_FEATURE_BINARY(Type, type_)                         \
          void VisitExpr_(const Type* op) final {                       \
            VisitExpr(op->a);                                           \
            x.push_back(global_table_.AffineOp[std::string(#type_)]);   \
            VisitExpr(op->b);                                           \
          }
          TVM_FEATURE_BINARY(AddNode, add);
          TVM_FEATURE_BINARY(SubNode, sub);
          TVM_FEATURE_BINARY(MulNode, mul);
          TVM_FEATURE_BINARY(DivNode, div);
          TVM_FEATURE_BINARY(ModNode, mod);
          TVM_FEATURE_BINARY(FloorDivNode, floordiv);
          TVM_FEATURE_BINARY(FloorModNode, floormod);
        #undef TVM_FEATURE_SIMPLE

        void VisitExpr_(const IntImmNode* int_) final{
          // hyper parameter: slog
          x.push_back((double)int_->value);
        }

        void VisitExpr_(const VarNode* var_) final{
          int64_t v = global_table_.getAffineLoopVar(var_);
          x.push_back((double)v);
        }

        TNGlobalTable global_table_;
        std::vector<double> x;
      };
      
      AffineCounter helper(*this);
      for(auto& region : block -> reads){
        for(auto& range : region -> region){
          helper(range->min);
        }
      }
      for(auto& region : block -> writes){
        for(auto& range : region->region){
          helper(range->min);
        }
      }
      affine_read_write.insert(affine_read_write.end(), helper.x.begin(), helper.x.end());
      addAffine(block, affine_read_write);
    }
/*************************************Affine*************************************/
    // src 2 dest
    // gloabl mem 1, cache/smem 2, reg 3
    std::pair<int64_t, int64_t> src2dest;
    int64_t blockcnt, curblockcnt;
};

class TNLoop{
public:
    TNLoop() = default;
    TNLoop(const ForNode* loop, TNGlobalTable& global_table){
        name_hint = loop->loop_var->name_hint;
        loopid = global_table.Loop2Loopid(loop);

        if(const int64_t* e = GetLoopIntExtent(loop)){
            extent = *e;
        }
        schedules.resize(2, 0);
        int64_t auto_unroll_attr = utils::TN::GetPragmaAutoUnroll(loop);
        if(auto_unroll_attr > 0){
            schedules[1] = auto_unroll_attr;
        }
        is_parallel = 1;
        if(loop->kind == ForKind::kSerial){
            schedules[0] = 0;
            is_parallel = 0;
        }
        else if(loop->kind == ForKind::kUnrolled){
            schedules[0] = 1;
            is_parallel = 0;
        }
        else if(loop->kind == ForKind::kVectorized){
            schedules[0] = 2;
        }
        else if(loop->kind == ForKind::kParallel){
            schedules[0] = 3;
        }
        else if(loop->kind == ForKind::kThreadBinding){
            std::string thread_tag = loop->thread_binding.value()->thread_tag;
            if (thread_tag == "blockIdx.x") {
                schedules[0] = 4;
            } else if (thread_tag == "blockIdx.y") {
                schedules[0] = 5;
            } else if (thread_tag == "blockIdx.z") {
                schedules[0] = 6;
            } else if (thread_tag == "threadIdx.x") {
                schedules[0] = 7;
            } else if (thread_tag == "threadIdx.y") {
                schedules[0] = 8;
            } else if (thread_tag == "threadIdx.z") {
                schedules[0] = 9;
            } else if (support::StartsWith(thread_tag, "vthread")) {
                schedules[0] = 10;
            } else {
                LOG(FATAL) << "ValueError: Unable to recognize thread tag: " << thread_tag;
            }
        }
        // std::cout << "construct TNLoop ..." << std::endl;
    }
    std::string name_hint = "";
    int64_t loopid = 0;  
    int64_t extent = 0;
    int64_t is_parallel = 0;    // 0 serial, 1 parallel
    // assert schedules.size == 2. 
    // [0],serial,para,vec,bindxyz,vthread,unroll 
    // [1], PragmaAutoUnroll
    std::vector<int64_t> schedules;

    void Export(std::shared_ptr<std::vector<double>> v) const {
        // hyper parameter slog
        double vs[] = {double(loopid), double(extent), double(is_parallel)};
        v->insert(v->end(), std::begin(vs), std::end(vs));
        v->insert(v->end(), schedules.begin(), schedules.end());
    }

    // void Export(std::vector<double>* v) const {
    //     double vs[] = {double(loopid), slog(double(extent)), double(is_parallel)};
    //     v->insert(v->end(), std::begin(vs), std::end(vs));
    //     v->insert(v->end(), schedules.begin(), schedules.end());
    // }
};

class TNLoopNest{
public:
    TNLoopNest() = default;
    TNLoopNest(const LoopNest& loopnest, TNGlobalTable& global_table){
        loop_nest.reserve(loopnest.loops.size());
        for(const auto& loop : loopnest.loops){
            loop_nest.push_back(std::make_shared<TNLoop>(loop, global_table));
        }

        // std::cout << "construct TNLoopNest ... nums:\t" << loopnest.loops.size() << std::endl;
    }
    // ~TNLoopNest(){
    //   for(TNLoop* ptr : loop_nest){
    //     delete ptr;
    //   }
    //   std::cout << "delete TNLoopNest" << std::endl;
    // }
    std::vector<std::shared_ptr<TNLoop>> loop_nest;

    void Export(std::shared_ptr<std::vector<double>> v) const {
        for(const auto& loop : loop_nest){
            loop->Export(v);
        }
    }
};


class TNBufferAccess{
public:
    TNBufferAccess() = default;
    TNBufferAccess(const LoopNest& loopnest, group1::Feature* f1, 
                group2::Feature* f2, TNGlobalTable& global_table):
    float_mad(f1->arith_ops.float_mad), float_add_sub(f1->arith_ops.float_add_sub),
    float_mul(f1->arith_ops.float_mul), float_div_mod(f1->arith_ops.float_div_mod),
    float_cmp(f1->arith_ops.float_cmp), float_math_func(f1->arith_ops.float_math_func),
    float_other_func(f1->arith_ops.float_other_func),
    int_mad(f1->arith_ops.int_mad), int_add_sub(f1->arith_ops.int_add_sub),
    int_mul(f1->arith_ops.int_mul), int_div_mod(f1->arith_ops.int_div_mod),
    int_cmp(f1->arith_ops.int_cmp), int_math_func(f1->arith_ops.int_math_func),
    int_other_func(f1->arith_ops.int_other_func),
    bool_op(f1->arith_ops.bool_op),
    select_op(f1->arith_ops.select_op)
    {
        // const auto& arith_ops = f1.arith_ops;
        // mad = arith_ops.float_mad + arith_ops.int_mad;
        // add_sub = arith_ops.float_add_sub + arith_ops.int_add_sub;
        // mul = arith_ops.float_mul + arith_ops.int_mul;
        // div_mod = arith_ops.float_div_mod + arith_ops.int_div_mod;
        // cmp = arith_ops.float_cmp + arith_ops.int_cmp;
        // math_func = arith_ops.float_math_func + arith_ops.int_math_func;
        // other_func = arith_ops.float_other_func + arith_ops.int_other_func;
        // bool_op = arith_ops.bool_op;
        // select_op = arith_ops.select_op;

        sub_buffer_features.reserve(3);
        for(int idx = 0; idx < (int)f2->sub_features.size(); ++idx){
            sub_buffer_features.emplace_back(loopnest, f2, idx);
        }

        // std::cout << "construct TNBufferAccess ..." << std::endl;

    }
    int64_t float_mad = 0;         // The number of float MAD (Multiply–add) ops
    int64_t float_add_sub = 0;     // The number of float add and sub ops
    int64_t float_mul = 0;         // The number of float multiply ops
    int64_t float_div_mod = 0;     // The number of float div and mod ops
    int64_t float_cmp = 0;         // The number of float comparison ops
    int64_t float_math_func = 0;   // The number of float math func calls
    int64_t float_other_func = 0;  // The number of other float func calls
    int64_t int_mad = 0;         // The number of integer MAD (Multiply–add) ops
    int64_t int_add_sub = 0;     // The number of integer add and sub ops
    int64_t int_mul = 0;         // The number of integer multiply ops
    int64_t int_div_mod = 0;     // The number of integer div and mod ops
    int64_t int_cmp = 0;         // The number of integer comparison ops
    int64_t int_math_func = 0;   // The number of integer math func calls
    int64_t int_other_func = 0;  // The number of other integer func calls
    int64_t bool_op = 0;    // The number of bool ops
    int64_t select_op = 0;  // The number of select ops
    // int64_t mad = 0;        // MAD (Multiply–add) ops
    // int64_t add_sub = 0;    // add and sub ops
    // int64_t mul = 0;        // multiply ops
    // int64_t div_mod = 0;    // div and mod ops
    // int64_t cmp = 0;        // comparison ops
    // int64_t math_func = 0;  // math func calls
    // int64_t other_func = 0; // float func calls
    // int64_t bool_op = 0;    // bool ops
    // int64_t select_op = 0;  // select ops

    // per buffer
    struct SubBufferFeature{
        SubBufferFeature() = default;
        SubBufferFeature(const LoopNest& loopnest, group2::Feature* f2, int idx):
        buffer(f2->sub_features[idx].buffer),
        loop_accessed_bytes(loopnest.loops.size()),
        access_shape(f2->sub_features[idx].access_shape),
        num_continuous_bytes(f2->sub_features[idx].num_continuous_bytes),
        min_stride(f2->sub_features[idx].min_stride),
        reuse_ct(f2->sub_features[idx].reuse_ct),
        bytes(f2->sub_features[idx].bytes),
        unique_bytes(f2->sub_features[idx].unique_bytes)
        { 
            ICHECK_EQ(loopnest.loops.size(), f2->sub_features[idx].loop_accessed_numel.size());
            int n_loops = loopnest.loops.size();
            for(int i = n_loops - 1; i >= 0; --i){
                loop_accessed_bytes[i][buffer] = \
                  f2->sub_features[idx].loop_accessed_numel[i][buffer] * buffer->dtype.bytes();
            }

        }
        const BufferNode* buffer = nullptr;
        std::vector<std::unordered_map<const BufferNode*, int64_t>> loop_accessed_bytes = {};
        std::vector<int64_t> access_shape;
        int64_t num_continuous_bytes = 1;
        int64_t min_stride = 0;
        int64_t reuse_ct = 0;
        double bytes;
        double unique_bytes;

        void Export(std::shared_ptr<std::vector<double> > v) const {
          // TNBufferAccess log, 
          double vs[] = {slog(double(num_continuous_bytes)), slog(double(min_stride)), 
                          slog(double(reuse_ct)),
                          slog(double(bytes)), slog(double(unique_bytes))};
          int n_loops = loop_accessed_bytes.size();
          std::vector<double> e1(n_loops);
          for(int i = 0; i < n_loops; ++i){
              // from outer2inner, align with TNLoopNest
              std::unordered_map<const BufferNode*, int64_t> m = loop_accessed_bytes[i];
              e1[i] = slog(double(m[buffer]));
          }
          int n_shapes = access_shape.size();
          std::vector<double> e2(n_shapes);
          for(int i = 0; i < n_shapes; ++i){
              e2[i] = slog(double(access_shape[i]));
          }
          v->insert(v->end(), std::begin(vs), std::end(vs));
          v->insert(v->end(), e1.begin(), e1.end());
          v->insert(v->end(), e2.begin(), e2.end());
        }
    };

    std::vector<SubBufferFeature> sub_buffer_features;

    void Export(std::shared_ptr<std::vector<double>> v) const {
      // TNBufferAccess log, 
      double vs[] = {slog(double(float_mad)),slog(double(float_add_sub)), slog(double(float_mul)),
                      slog(double(float_div_mod)),slog(double(float_cmp)), slog(double(float_math_func)),
                      slog(double(float_other_func)),
                      slog(double(int_mad)),slog(double(int_add_sub)), slog(double(int_mul)),
                      slog(double(int_div_mod)),slog(double(int_cmp)), slog(double(int_math_func)),
                      slog(double(int_other_func)),
                      slog(double(bool_op)),slog(double(select_op))};
      v->insert(v->end(), std::begin(vs), std::end(vs));
      for(const auto& sub_buffer_feature : sub_buffer_features){
          sub_buffer_feature.Export(v);
      }
    }
};

class TNIntrinAccess{
public:
    TNIntrinAccess() = default;
    TNIntrinAccess(std::string IntrinOp, TNGlobalTable& global_table_){
      _export.clear();
      _export.reserve(15 + 16 + 10 * 3);
      // _export = global_table_.GetIntrin(IntrinOp);

      std::vector<double> v = global_table_.GetIntrin(IntrinOp);
      _export.insert(_export.end(), v.begin(), v.end());
      // std::cout << "construct TNIntrinAccess" << std::endl;
    }
    void Export(std::shared_ptr<std::vector<double>> v) const {
      v->insert(v->end(), _export.begin(), _export.end());
    }

    std::vector<double> _export;

};
class TNBehavior{
public:
    // is_intrin = 0
    TNBehavior(std::string type_, int64_t is_intrin_,
        const LoopNest& loopnest_, TNGlobalTable& global_table_,
        group1::Feature* f1_, group2::Feature* f2_):
    loop_nest(std::make_shared<TNLoopNest>(loopnest_, global_table_)),
    buffer_access(std::make_shared<TNBufferAccess>(loopnest_, f1_, f2_, global_table_))
    { 
        intrin_access = nullptr;
        set_behavior_attrs(type_, is_intrin_);
        _export(global_table_);

        // std::cout << "construct TNBehavior is_intrin = 0 ..." << std::endl;
        
    };

    // is_intrin = 1
    TNBehavior(std::string type_, int64_t is_intrin_,
        const LoopNest& loopnest_, TNGlobalTable& global_table_,
        std::string IntrinOp):
    loop_nest(std::make_shared<TNLoopNest>(loopnest_, global_table_)),
    intrin_access(std::make_shared<TNIntrinAccess>(IntrinOp, global_table_))
    {   
        buffer_access = nullptr;
        set_behavior_attrs(type_, is_intrin_);
        _export(global_table_);
        // std::cout << "construct TNBehavior is_intrin = 1 ..." << std::endl;
    };

    // ~TNBehavior(){
    //   delete loop_nest;
    //   delete buffer_access;
    //   delete intrin_access;
    //   std::cout << "delete TNBehavior" << std::endl;
    // }

    void set_behavior_attrs(std::string type_, int64_t is_intrin_){
        is_intrin = is_intrin_;
        n_behavior = 5;
        one_hot.reserve(n_behavior);
        one_hot.resize(n_behavior, 0.);
        if(type_ == "Preprocess"){
            behavior_id = 0;
        }
        if(type_ == "Load"){
            behavior_id = 1;
        }
        else if(type_ == "Compute"){
            behavior_id = 2;
        }
        else if(type_ == "Store"){
            behavior_id = 3;
        }
        else if(type_ == "Epilogue"){
            behavior_id = 4;
        }
        one_hot[behavior_id] = 1.;
        embed_size = 179;
        loop_embedding = std::make_shared<std::vector<double>>();
        affine_embedding = std::make_shared<std::vector<double>>();
        access_embedding = std::make_shared<std::vector<double>>(); 
        loop_embedding->reserve(embed_size);
        affine_embedding->reserve(embed_size);
        access_embedding->reserve(embed_size);
    }
    int n_behavior;
    int behavior_id;
    std::vector<double> one_hot;
    int64_t is_intrin;
    // std::vector<double> loop_embedding;
    // std::vector<double> affine_embedding;
    // std::vector<double> access_embedding;

    std::shared_ptr<std::vector<double> > loop_embedding, affine_embedding, access_embedding; 
    int embed_size;

    std::shared_ptr<TNLoopNest> loop_nest;
    std::shared_ptr<TNBufferAccess> buffer_access;
    std::shared_ptr<TNIntrinAccess> intrin_access;

    void _export(TNGlobalTable& global_table_){
        loop_embedding->insert(loop_embedding->end(), one_hot.begin(), one_hot.end());
        // loop_embedding->push_back(double(is_intrin));
        access_embedding->insert(access_embedding->end(), one_hot.begin(), one_hot.end());
        // access_embedding->push_back(double(is_intrin));
        affine_embedding->insert(affine_embedding->end(), one_hot.begin(), one_hot.end());
        // affine_embedding->push_back(double(is_intrin));
        loop_nest->Export(loop_embedding);
        // Affine Embedding
        std::vector<double> affine_read_write_ = global_table_.getAffine();
        affine_embedding->insert(affine_embedding->end(), affine_read_write_.begin(), affine_read_write_.end());
        // Affine Embedding
        if(!is_intrin){
            buffer_access->Export(access_embedding);
        }
        else{
            intrin_access->Export(access_embedding);
        }
        // std::cout << "enter___checkAffine___" << std::endl;
        // for(const auto& it : affine_embedding) std::cout << it << " ";
        // std::cout << std::endl;
        // std::cout << affine_embedding.size();
        // std::cout << "exit___checkAffine___" << std::endl;

        loop_embedding->insert(loop_embedding->begin() + 5, (double)is_intrin);
        loop_embedding->insert(loop_embedding->begin() + 6, (double)global_table_.src2dest.first);
        loop_embedding->insert(loop_embedding->begin() + 7, (double)global_table_.src2dest.second);
        // std::cout << "loop_embedding.size():\t" << loop_embedding->size() << std::endl;
        loop_embedding->resize(embed_size, 0.);

        // (*loop_embedding)[embed_size - 3] = (double)is_intrin;
        // (*loop_embedding)[embed_size - 2] = (double)global_table_.src2dest.first;
        // (*loop_embedding)[embed_size - 1] = (double)global_table_.src2dest.second;

        affine_embedding->insert(affine_embedding->begin() + 5, (double)is_intrin);
        affine_embedding->insert(affine_embedding->begin() + 6, (double)global_table_.src2dest.first);
        affine_embedding->insert(affine_embedding->begin() + 7, (double)global_table_.src2dest.second);
        // std::cout << "affine_embedding.size():\t" << affine_embedding->size() << std::endl;

        affine_embedding->resize(embed_size, 0.);
        // (*affine_embedding)[embed_size - 3] = (double)is_intrin;
        // (*affine_embedding)[embed_size - 2] = (double)global_table_.src2dest.first;
        // (*affine_embedding)[embed_size - 1] = (double)global_table_.src2dest.second;

        access_embedding->insert(access_embedding->begin() + 5, (double)is_intrin);
        access_embedding->insert(access_embedding->begin() + 6, (double)global_table_.src2dest.first);
        access_embedding->insert(access_embedding->begin() + 7, (double)global_table_.src2dest.second);
        // std::cout << "access_embedding.size() \t"<< access_embedding->size() << std::endl;
        access_embedding->resize(embed_size, 0.);
        // (*access_embedding)[embed_size - 3] = (double)is_intrin;
        // (*access_embedding)[embed_size - 2] = (double)global_table_.src2dest.first;
        // (*access_embedding)[embed_size - 1] = (double)global_table_.src2dest.second;
    }
    void Export(std::vector<std::vector<double>>* v) const{
        v->push_back(*loop_embedding);
        v->push_back(*affine_embedding);
        v->push_back(*access_embedding);
        // std::cout << "enter___checkAffine___" << std::endl;
        // for(const auto& it : affine_embedding) std::cout << it << " ";
        // std::cout << std::endl;
        // std::cout << "exit___checkAffine___" << std::endl;
    }
};

class TNProgram{
public:
    TNProgram(){
        program.reserve(27);
        seq_len = 27;   
    }
    // ~TNProgram(){
    //   for(TNBehavior* ptr : program){
    //     delete ptr;
    //   }
    //   std::cout << "delete TNProgram" << std::endl;
    // }
    void add_behavior(std::shared_ptr<TNBehavior> b){
        program.push_back(b);
    }
    std::vector<std::shared_ptr<TNBehavior>> program;
    int seq_len;

    void Export(std::vector<std::vector<double>>* v) const{
        for(const auto& b : program){
            b->Export(v);
        }
        int embed_size = (*v)[0].size();
        int cur_seq_len = (*v).size();
        std::vector<double> pad_seq(embed_size, 0.);
        for(int i = cur_seq_len; i < seq_len; ++i){
            v->push_back(pad_seq);
        }
    }
};

class FindIntrinFromBufferStoreNode : public ExprVisitor{
public:
  FindIntrinFromBufferStoreNode():IntrinOp(""), has_intrin(false){};
  void VisitExpr_(const CallNode* op) final {
    if(op->op.same_as(builtin::call_llvm_pure_intrin())){
        // for(const auto& item : op->args){
        //   std::cout << item << std::endl;
        // }

        // find avx512
        if(Downcast<IntImm>(op->args[0])->value == 9924){
          // std::cout << "find avx512" << std::endl;
          IntrinOp = "dot_16x4_avx512";
          has_intrin = true;
        }
        // find vnni
        else if(Downcast<IntImm>(op->args[0])->value == 10060){
          // std::cout << "find vnni" << std::endl;
          IntrinOp = "dot_16x4_vnni";
          has_intrin = true;
        }

        // find neon
        else if(Downcast<IntImm>(op->args[0])->value == 431){
          // std::cout << "find neon" << std::endl;
          IntrinOp = "dot_4x4_i8i8s32_neon";
          has_intrin = true;

        }
        // find sdot
        else if(Downcast<IntImm>(op->args[0])->value == 507){
          // std::cout << "find sdot" << std::endl;
          IntrinOp = "dot_4x4_i8i8s32_sdot";
          has_intrin = true;
        }
    }
  }
  std::string IntrinOp;
  bool has_intrin;
};


/*********Tensorize-Notation*********/

class BlockCountHelper : public StmtVisitor{
public:
  BlockCountHelper():blockcnt(0){};
  int64_t blockcnt;
  void VisitStmt_(const BlockNode* block) final {
      if(std::string(block->name_hint).find("relu") == std::string::npos &&
      std::string(block->name_hint).find("add") == std::string::npos){
        blockcnt += 1;
      }
       
      // std::cout << block->name_hint << std::endl;
      StmtVisitor::VisitStmt_(block);
  }
};

/*! \brief The main feature extractor */
class PerBlockFeatureCollector : private StmtVisitor {
 public:
  static tir::TN::TNProgram Collect(bool is_gpu, int64_t cache_line_bytes,
                                      int64_t arith_intensity_curve_num_samples,
                                      const IRModule& mod) {
    PerBlockFeatureCollector collector(is_gpu, cache_line_bytes, arith_intensity_curve_num_samples);
    for (const auto& kv : mod->functions) {
      if (const PrimFuncNode* func = kv.second.as<PrimFuncNode>()) {
        // blockcnt
        BlockCountHelper blockcnthelper;
        blockcnthelper(func->body);
        collector.global_table_.blockcnt = blockcnthelper.blockcnt;
        // std:: cout << "blockcnt:\t" << collector.global_table_.blockcnt << std::endl;
        // blockcnt
        collector(func->body);
        // for (const auto& it : func->buffer_map) {
        //   collector.HandleBufferAlloc(it.second);
        // }
      }
    }
    // std::vector<Feature> result;
    // result.reserve(collector.buffer_features_.size());
    // for (auto& it : collector.buffer_features_) {
    //   Feature& feature = it.second;
    //   if (feature.buffer != nullptr) {
    //     ICHECK(feature.group1);
    //     ICHECK(feature.group2);
    //     ICHECK(feature.group3);
    //     ICHECK(feature.group5);
    //     if (feature.group4 == nullptr) {
    //       feature.group4 = std::make_shared<group4::Feature>();
    //     }
    //     result.push_back(std::move(feature));
    //   }
    // }
    // std::sort(result.begin(), result.end());
    // return result;
    return collector.TensorizedProgram_;
  }

 private:
  void VisitStmt_(const ForNode* loop) final {
    int64_t auto_unroll;
    ForVec* for_vec = loop_nest_.Push(loop, &auto_unroll);
    global_table_.AddLoop(loop);
    StmtVisitor::VisitStmt_(loop);
    loop_nest_.Pop(loop, for_vec, auto_unroll);
    global_table_.DelLoop(loop);
  }

  void VisitStmt_(const BufferStoreNode* store) final {
    // if (store->value->IsInstance<IntImmNode>() || store->value->IsInstance<FloatImmNode>()) {
    //   return;
    // }
    // affine
    // std::cout << "__enter__BufferStoreNode__" << std::endl;
    // std::cout << GetRef<BufferStore>(store) << std::endl << std::endl;
    // std::cout << "__exit__BufferStoreNode__" << std::endl;
    global_table_.MakeAffine();

    const BufferNode* buffer = store->buffer.get();
    Feature& feature = buffer_features_[buffer];
    if (feature.buffer == nullptr) {
      feature.buffer = buffer;
      feature.buffer_order = buffer_features_.size();
    }
    feature.group1 = std::make_shared<group1::Feature>(store, loop_nest_, is_gpu_);
    feature.group2 =
        std::make_shared<group2::Feature>(store, loop_nest_, cache_line_bytes_, &for_touched_bytes_,
                                          &buffer_touched_under_loop_, &analyzer_);
    // feature.group3 =
    //     std::make_shared<group3::Feature>(arith_intensity_curve_num_samples_, loop_nest_,
    //                                       for_touched_bytes_, feature.group1->arith_ops);
    // feature.group5 = std::make_shared<group5::Feature>(loop_nest_);
    // TN
    tir::TN::FindIntrinFromBufferStoreNode find_intrin_helper;
    find_intrin_helper(store->value);
    if(find_intrin_helper.has_intrin){
      global_table_.IntrinCompute += 1;
      global_table_.src2dest.first = 1; global_table_.src2dest.second = 3; // global 2 reg
      TensorizedProgram_.add_behavior(std::make_shared<TNBehavior>("Compute", 1, loop_nest_, global_table_, find_intrin_helper.IntrinOp));
      
    }
    std::string type_ = global_table_.GetGeneralBehavior();
    TensorizedProgram_.add_behavior(std::make_shared<TNBehavior>(type_, 0, loop_nest_, global_table_, 
                                             feature.group1.get(), feature.group2.get()));
  }

  void VisitExpr(const PrimExpr& e) {
    if(e->IsInstance<CallNode>()){
      
      // std::cout << "__enter__PrimExpr__" << std::endl;
      // std::cout << (e) << std::endl;
      // std::cout << "__exit__PrimExpr__" << std::endl;

      Call op = Downcast<Call>(e);
      if(op->op.same_as(builtin::tvm_fill_fragment())){
        // affine
        global_table_.MakeAffine();
        global_table_.tccurmatrix = 'c';
        global_table_.tcname = "wmma";
        global_table_.tctype_ = "fill";
        int s1 = Downcast<IntImm>(op->args[1])->value;
        int s2 = Downcast<IntImm>(op->args[2])->value;
        int s3 = Downcast<IntImm>(op->args[3])->value;
        global_table_.tcshape = \
          std::to_string(s1) + "x" + std::to_string(s2) + "x" + std::to_string(s3);
        global_table_.tcmatrixcdtype = \
          global_table_.GetBufferVarDtype(std::string(Downcast<Var>(op->args[0])->name_hint));
        std::string IntrinOp = \
            global_table_.tcname + "_" 
          + global_table_.tctype_ + "_" 
          + global_table_.tcshape + "_"
          + global_table_.tcmatrixcdtype;
        global_table_.IntrinPreprocess += 1;
        global_table_.src2dest.first = global_table_.src2dest.second = 3; // reg 2 reg
        TensorizedProgram_.add_behavior(std::make_shared<TNBehavior>("PreProcess", 1, loop_nest_, global_table_, IntrinOp));
        // std::cout << IntrinOp << std::endl;
      }
      if(op->op.same_as(builtin::tvm_load_matrix_sync())){
        // affine
        global_table_.MakeAffine();
        std::string* curmatrixdtype = nullptr;
        std::string* curmatrixlayout = nullptr;
        if(!global_table_.IntrinLoad){
          global_table_.tccurmatrix = "a";
          curmatrixdtype = &(global_table_.tcmatrixadtype);
          curmatrixlayout = &(global_table_.tcmatrixalayout);
        }
        else{
          global_table_.tccurmatrix = "b";
          curmatrixdtype = &(global_table_.tcmatrixbdtype);
          curmatrixlayout = &(global_table_.tcmatrixblayout);
        }
        
        global_table_.tcname = "wmma";
        global_table_.tctype_ = "load";

        int s1 = Downcast<IntImm>(op->args[1])->value;
        int s2 = Downcast<IntImm>(op->args[2])->value;
        int s3 = Downcast<IntImm>(op->args[3])->value;
         global_table_.tcshape = \
          std::to_string(s1) + "x" + std::to_string(s2) + "x" + std::to_string(s3);
        
        std::string layout_ = std::string(op->args[7].as<StringImmNode>()->value);
        if(layout_ == "row_major"){
          *curmatrixlayout = "";
        }
        else{
          *curmatrixlayout = "trans";
        }
        
        *curmatrixdtype = \
          global_table_.GetBufferVarDtype(std::string(Downcast<Var>(op->args[0])->name_hint));
                std::string IntrinOp = \
            global_table_.tcname + "_" 
          + global_table_.tctype_ + "_" 
          + global_table_.tcshape + "_"
          + *curmatrixdtype + "_" 
          + global_table_.tccurmatrix + "_"
          + ((*curmatrixlayout).empty() ? ("") : (*curmatrixlayout + "_"))
          + "shared_dyn";
        curmatrixdtype = nullptr;
        curmatrixlayout = nullptr;
        global_table_.IntrinLoad += 1;
        global_table_.src2dest.first = 2; global_table_.src2dest.second = 3; // smem 2 reg
        TensorizedProgram_.add_behavior(std::make_shared<TNBehavior>("Load", 1, loop_nest_, global_table_, IntrinOp));
        // std::cout << IntrinOp << std::endl;
      }
      if(op->op.same_as(builtin::tvm_mma_sync())){
        // affine
        global_table_.MakeAffine();
        global_table_.tcname = "wmma";
        global_table_.tctype_ = "sync";
        std::string IntrinOp = \
            global_table_.tcname + "_" 
          + global_table_.tctype_ + "_" 
          + global_table_.tcshape + "_"
          + global_table_.tcmatrixadtype 
          + global_table_.tcmatrixbdtype 
          + global_table_.tcmatrixcdtype
          + (!global_table_.tcmatrixalayout.empty() || !global_table_.tcmatrixblayout.empty() ?
              "_trans" : "");
        global_table_.IntrinCompute += 1; 
        global_table_.src2dest.first = global_table_.src2dest.second = 3; // reg 2 reg
        TensorizedProgram_.add_behavior(std::make_shared<TNBehavior>("Compute", 1, loop_nest_, global_table_, IntrinOp));
        // std::cout << IntrinOp << std::endl;
      }
      if(op->op.same_as(builtin::tvm_store_matrix_sync())){
        // affine
        global_table_.MakeAffine();
        global_table_.tcname = "wmma";
        global_table_.tctype_ = "store";
        std::string IntrinOp = \
            global_table_.tcname + "_" 
          + global_table_.tctype_ + "_" 
          + global_table_.tcshape + "_"
          + global_table_.tcmatrixcdtype + "_"
          + "shared_dyn";
        global_table_.IntrinStore += 1;
        global_table_.src2dest.first = 3; global_table_.src2dest.second = 2; // reg 2 smem
        TensorizedProgram_.add_behavior(std::make_shared<TNBehavior>("Store", 1, loop_nest_, global_table_, IntrinOp));
        // std::cout << IntrinOp << std::endl;
      }
      if(op->op.same_as(builtin::call_llvm_pure_intrin())){
        // NOUSE
      }
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    for (const Buffer& buffer : block->alloc_buffers){
      std::string dtype;
      if(buffer->dtype.is_float16()){
        dtype = "f16";
      }
      else{
        dtype = "f32";
      }
      global_table_.buffer_data_dtype[std::string(buffer->data->name_hint)] = dtype;
    }

    // std::cout << "enter___read__write__inblock__"<< std::endl;

    // for(auto& region : block -> reads){
    //   for(auto& range : region->region){
    //     std::cout << range -> min << " " <<range->extent << std::endl; 
    //   }
    // }

    // std::cout << "exit___read__write__inblock__" << std::endl;


    /*************************************Affine*************************************/
    global_table_.CurBlock_affinehelper = block;
    
    /*************************************Affine*************************************/

    //src 2 dest blockcnt
    global_table_.curblockcnt += 1;
    //src 2 dest blockcnt
    StmtVisitor::VisitStmt_(block);
    // for (const Buffer& buffer : block->alloc_buffers) {
    //   HandleBufferAlloc(buffer);
    // }
  }

  void HandleBufferAlloc(const Buffer& buffer) {
    Feature& feature = buffer_features_[buffer.get()];
    feature.group4 = std::make_shared<group4::Feature>(loop_nest_, buffer, &analyzer_);
  }

  explicit PerBlockFeatureCollector(bool is_gpu, int64_t cache_line_bytes,
                                    int64_t arith_intensity_curve_num_samples)
      : is_gpu_(is_gpu),
        cache_line_bytes_(cache_line_bytes),
        arith_intensity_curve_num_samples_(arith_intensity_curve_num_samples) {}

  bool is_gpu_;
  int64_t cache_line_bytes_;
  int64_t arith_intensity_curve_num_samples_;
  arith::Analyzer analyzer_;
  LoopNest loop_nest_ = {};
  IntVec for_touched_bytes_ = {};
  ForBufferMap<IntVec> buffer_touched_under_loop_ = {};
  std::unordered_map<const BufferNode*, Feature> buffer_features_ = {};
  TNProgram TensorizedProgram_;
  TNGlobalTable global_table_;
};



}  // namespave TN
}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

class PerBlockFeatureNode : public FeatureExtractorNode {
 public:
  int buffers_per_store;
  int arith_intensity_curve_num_samples;
  int cache_line_bytes;
  bool extract_workload;
  int feature_vector_length;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("buffers_per_store", &buffers_per_store);
    v->Visit("arith_intensity_curve_num_samples", &arith_intensity_curve_num_samples);
    v->Visit("cache_line_bytes", &cache_line_bytes);
    v->Visit("feature_vector_length", &feature_vector_length);
  }

  void ExtractSingle(IRModule mod, bool is_gpu, std::vector<std::vector<double>>* results) {
    static transform::Sequential passes = tir::transform::TN::PassListForPerBlockFeature();
    mod = passes(std::move(mod));
    // std::ofstream outfile("/home/zhwang/trash.print/passedmod1");
    // outfile << mod << std::endl;
    // std::vector<tir::TN::Feature> features = tir::TN::PerBlockFeatureCollector::Collect(
    //     is_gpu, this->cache_line_bytes, this->arith_intensity_curve_num_samples, mod);
    // int n_features = features.size();
    // results->resize(n_features);
    // for (int i = 0; i < n_features; ++i) {
    //   const tir::TN::Feature& feature = features[i];
    //   std::vector<double>& result = (*results)[i];
    //   result.reserve(feature_vector_length);
    //   feature.group1->Export(&result);
    //   feature.group2->Export(&result, this->buffers_per_store);
    //   feature.group3->Export(&result);
    //   feature.group4->Export(&result, feature.group5->outer_prod);
    //   feature.group5->Export(&result);
    // }

    tir::TN::TNProgram program = tir::TN::PerBlockFeatureCollector::Collect(
        is_gpu, this->cache_line_bytes, this->arith_intensity_curve_num_samples, mod);
    results->reserve(program.seq_len);
    program.Export(results);
  }

  Array<runtime::NDArray> ExtractFrom(const TuneContext& tune_context,
                                      const Array<MeasureCandidate>& candidates) {
    bool is_gpu = tune_context->target.value()->kind->name == "cuda";
    std::vector<runtime::NDArray> results;
    results.resize(candidates.size());
    std::shared_ptr<tir::TN::group6::Feature> feature_group6 = nullptr;
    if (extract_workload) {
      feature_group6 = std::make_shared<tir::TN::group6::Feature>(tune_context->mod.value());
    }
    auto f = [this, is_gpu, &feature_group6, &candidates, &results](int, int task_id) -> void {
      const auto& candidate = candidates[task_id];
      std::vector<std::vector<double>> features;
      ExtractSingle(DeepCopyIRModule(candidate->sch->mod()), is_gpu, &features);
      if (extract_workload) {
        for (auto& feature : features) {
          feature_group6->Export(&feature);
        }
      }
      results[task_id] = tir::utils::TN::AsNDArray(features, this->feature_vector_length);
    };
    support::parallel_for_dynamic(0, candidates.size(), tune_context->num_threads, f);
    return results;
  }

  static constexpr const char* _type_key = "meta_schedule.PerBlockFeature";
  TVM_DECLARE_FINAL_OBJECT_INFO(PerBlockFeatureNode, FeatureExtractorNode);
};

FeatureExtractor FeatureExtractor::PerBlockFeature(int buffers_per_store,
                                                   int arith_intensity_curve_num_samples,
                                                   int cache_line_bytes, bool extract_workload) {
  ObjectPtr<PerBlockFeatureNode> n = make_object<PerBlockFeatureNode>();
  n->buffers_per_store = buffers_per_store;
  n->arith_intensity_curve_num_samples = arith_intensity_curve_num_samples;
  n->cache_line_bytes = cache_line_bytes;
  n->extract_workload = extract_workload;
  n->feature_vector_length = tir::TN::group1::Feature::kCount +                                  //
                             tir::TN::group2::Feature::SubFeature::kCount * buffers_per_store +  //
                             arith_intensity_curve_num_samples +                             //
                             tir::TN::group4::Feature::kCount +                                  //
                             tir::TN::group5::Feature::kCount;
  if (extract_workload) {
    n->feature_vector_length += tir::TN::group6::Feature::kCount;
  }
  return FeatureExtractor(n);
}

TVM_REGISTER_NODE_TYPE(PerBlockFeatureNode);
TVM_REGISTER_GLOBAL("meta_schedule.FeatureExtractorPerBlockFeature")
    .set_body_typed(FeatureExtractor::PerBlockFeature);

}  // namespace meta_schedule
}  // namespace tvm



