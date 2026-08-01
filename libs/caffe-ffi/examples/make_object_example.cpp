/**
 * @file make_object_example.cpp
 * @brief 迁移指南步骤 4 的完整示例：从 std::shared_ptr 参数传递迁移到 ObjectPtr 风格
 *
 * 本文件演示了迁移指南（SHARED_PTR_TO_INTRUSIVE_REFCOUNT_MIGRATION.md）中
 * 步骤 4 的完整用法，覆盖三种场景：
 *   1. 内部函数传递（裸指针，零开销）
 *   2. 所有权转移（ObjectPtr 值传递，原子 increment）
 *   3. FFI 边界（ObjectPtr& + lambda 适配）
 *
 * 编译方法（项目根目录）：
 *   cmake --build build --target make_object_example
 * 运行：
 *   ./build/bin/make_object_example
 */

#include <tvm/ffi/container/array.h>  // make_object, Array<>
#include <tvm/ffi/function.h>         // TVM_FFI_REGISTER_OBJECT
#include <tvm/ffi/object.h>           // Object, ObjectPtr, GetRef
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// 1. 模拟一个业务对象：Weight（权重张量）
// ============================================================================

namespace myproject {

using namespace tvm::ffi;

/**
 * @brief 权重对象：封装一个浮点数组及其名称
 *
 * 继承 Object 获得侵入式引用计数，支持 make_object 创建、
 * GetRef 从裸指针恢复、ObjectPtr 智能指针管理。
 */
class Weight : public Object {
 public:
  static constexpr bool _type_mutable = true;

  /// 构造函数：分配 n 个 float，初始化为 0
  explicit Weight(std::string name, int64_t n)
      : name_(std::move(name)), size_(n), data_(new float[n]()) {
    std::cout << "[Weight] Created '" << name_ << "' with " << size_
              << " elements\n";
  }

  ~Weight() override {
    std::cout << "[Weight] Destroyed '" << name_ << "'\n";
    delete[] data_;
  }

  const std::string& name() const { return name_; }
  int64_t size() const { return size_; }

  /// 读取第 i 个元素（const 版本，只读访问）
  float Get(int64_t i) const {
    return (i >= 0 && i < size_) ? data_[i] : 0.0f;
  }

  /// 设置第 i 个元素（非 const 版本，可写访问）
  void Set(int64_t i, float v) {
    if (i >= 0 && i < size_) data_[i] = v;
  }

  /// 获取裸数据指针（只读）
  const float* data() const { return data_; }

  /// 获取裸数据指针（可写）---- 如果支持 COW，这里会触发克隆
  float* mutable_data() { return data_; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("myproject.Weight", Weight, Object);

 private:
  std::string name_;
  int64_t size_;
  float* data_;
};

using WeightPtr = ObjectPtr<Weight>;

// FFI 类型注册
TVM_FFI_REGISTER_OBJECT(Weight)
    .def("name", &Weight::name)
    .def("size", &Weight::size)
    .def("get", &Weight::Get)
    .def("set", &Weight::Set);

// ============================================================================
// 2. 场景一：内部函数传递（裸指针，零开销，不改变 refcount）
// ============================================================================

/// 只读访问：传 const Weight*，不增加 refcount
void PrintWeightInfo(const Weight* w) {
  if (w == nullptr) {
    std::cout << "  PrintWeightInfo: nullptr\n";
    return;
  }
  std::cout << "  PrintWeightInfo: name=" << w->name() << ", size=" << w->size()
            << ", first=" << w->Get(0) << "\n";
}

/// 修改权重：传 Weight*，不增加 refcount
void ScaleWeight(Weight* w, float factor) {
  if (w == nullptr) return;
  float* data = w->mutable_data();
  for (int64_t i = 0; i < w->size(); ++i) {
    data[i] *= factor;
  }
  std::cout << "  ScaleWeight: scaled '" << w->name() << "' by " << factor << "\n";
}

// ============================================================================
// 3. 场景二：所有权转移（ObjectPtr 值传递，原子 increment）
// ============================================================================

/// 存储所有权：按值接收 ObjectPtr（拷贝 = 原子 increment）
class WeightRegistry {
 public:
  void Register(WeightPtr w) {
    // w 是值传递的拷贝，refcount 已经 +1
    std::cout << "  WeightRegistry: registered '" << w->name() << "'\n";
    weights_.push_back(std::move(w));  // move 零开销，refcount 不变化
  }

  void PrintAll() const {
    std::cout << "  WeightRegistry::PrintAll: " << weights_.size() << " weights\n";
    for (const auto& w : weights_) {
      PrintWeightInfo(w.get());  // 内部调用：裸指针
    }
  }

  size_t Count() const { return weights_.size(); }

 private:
  std::vector<WeightPtr> weights_;
};

// ============================================================================
// 4. 场景三：FFI 边界（ObjectPtr& + lambda 适配）
// ============================================================================

/// 模拟 FFI 注册：用 lambda 将 ObjectPtr 参数转为裸指针
class WeightProcessor {
 public:
  static void ProcessWeight(WeightPtr w) {
    // FFI 调用方传入 ObjectPtr，内部转为裸指针
    PrintWeightInfo(w.get());  // 零开销转换
    ScaleWeight(w.get(), 2.0f);
  }

  static void CompareWeights(const WeightPtr& a, const WeightPtr& b) {
    // const ObjectPtr& 接收，不增加 refcount
    PrintWeightInfo(a.get());
    PrintWeightInfo(b.get());
    if (a->data() == b->data()) {
      std::cout << "  CompareWeights: same buffer (shared)\n";
    } else {
      std::cout << "  CompareWeights: different buffers\n";
    }
  }
};

// ============================================================================
// 5. 完整演示
// ============================================================================

}  // namespace myproject

int main() {
  using namespace myproject;
  using namespace tvm::ffi;

  std::cout << "========== make_object 创建示例 ==========\n\n";

  // ── 步骤 3：make_object 创建（等价于 std::make_shared） ──
  auto w1 = make_object<Weight>("layer1_weights", 256);
  w1->Set(0, 3.14f);
  std::cout << "\n";

  // ── 场景一：内部函数传递（裸指针） ──
  std::cout << "--- 场景一：裸指针传递（零开销） ---\n";
  PrintWeightInfo(w1.get());  // 不增加 refcount
  ScaleWeight(w1.get(), 2.0f);
  PrintWeightInfo(w1.get());
  std::cout << "\n";

  // ── 场景二：所有权转移（ObjectPtr 值传递） ──
  std::cout << "--- 场景二：所有权转移（ObjectPtr 值传递） ---\n";
  {
    WeightRegistry registry;
    registry.Register(w1);  // 拷贝 w1 → refcount 变为 2
    auto w2 = make_object<Weight>("layer2_weights", 128);
    w2->Set(0, 42.0f);
    registry.Register(w2);  // 拷贝 w2 → refcount 变为 2
    registry.PrintAll();
    std::cout << "  Registry count: " << registry.Count() << "\n";
    // registry 离开作用域 → 其持有的 ObjectPtr 析构，refcount 各减 1
  }
  std::cout << "\n";

  // ── 场景三：FFI 边界（lambda 适配） ──
  std::cout << "--- 场景三：FFI 边界（lambda 适配） ---\n";
  WeightProcessor::ProcessWeight(w1);
  std::cout << "\n";

  // ── 零拷贝共享演示 ──
  std::cout << "--- 零拷贝共享演示 ---\n";
  auto w3 = make_object<Weight>("shared_source", 10);
  w3->Set(0, 99.9f);

  // 模拟 ShareData：同一裸指针被两个 ObjectPtr 持有
  // 注意：真正的 ShareData 通过 Tensor 赋值实现，此处简化为 ObjectPtr 别名
  WeightPtr w4 = w3;  // 拷贝 ObjectPtr → refcount 变为 2
  WeightProcessor::CompareWeights(w3, w4);
  std::cout << "\n";

  // ── 从裸指针安全恢复（步骤 5） ──
  std::cout << "--- 从裸指针安全恢复（步骤 5） ---\n";
  {
    Weight* raw = w3.get();
    WeightPtr recovered = GetRef<WeightPtr>(raw);  // 安全：refcount++
    PrintWeightInfo(recovered.get());
    std::cout << "  recovered 离开作用域 → refcount--\n";
  }
  std::cout << "\n";

  std::cout << "========== 演示结束 ==========\n";
  // w1, w3, w4 析构 → refcount 归零时自动释放内存
  return 0;
}