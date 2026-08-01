#include "test_harness.hpp"

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/common.hpp"

#include <vector>

using namespace caffe_ffi;

// ============================================================================
// ObjectPtrMigration 测试套件
// 覆盖迁移指南（SHARED_PTR_TO_INTRUSIVE_REFCOUNT_MIGRATION.md）中
// 场景二（所有权转移）和场景三（FFI 边界/裸指针恢复）的核心模式
// ============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// 场景二：ObjectPtr 所有权转移（WeightRegistry 模式）
// 覆盖：拷贝构造、移动语义、容器存储、生命周期安全
// ─────────────────────────────────────────────────────────────────────────────

/// 验证 ObjectPtr 拷贝构造 = 共享所有权（refcount 递增）
TEST(ObjectPtrMigration, CopyIncreasesRefcount) {
  auto obj = make_object<Blob>(std::vector<int64_t>{2, 3});
  obj->cpu_mutable_data()[0] = 42.0f;

  // 拷贝构造 → refcount 变为 2
  ObjectPtr<Blob> copy = obj;

  // 指向同一个对象
  EXPECT_EQ(obj.get(), copy.get());
  EXPECT_EQ(obj->cpu_data(), copy->cpu_data());

  // 通过任一指针写入，另一方可见
  copy->cpu_mutable_data()[1] = 99.0f;
  EXPECT_NEAR(obj->cpu_mutable_data()[1], 99.0f, 1e-5f);
}

/// 验证容器持有 ObjectPtr 时，原始指针离开作用域后数据仍有效
/// 对应 WeightRegistry::Register() 后原始 owner 析构的场景
TEST(ObjectPtrMigration, RegistryHoldsOwnershipAfterSourceOutOfScope) {
  std::vector<ObjectPtr<Blob>> registry;

  {
    auto obj = make_object<Blob>(std::vector<int64_t>{4});
    obj->cpu_mutable_data()[0] = 99.0f;
    obj->cpu_mutable_data()[1] = 88.0f;
    registry.push_back(obj);  // 拷贝 → refcount 变为 2
    EXPECT_EQ(obj.get(), registry.back().get());
  }
  // obj 离开作用域，refcount 降为 1，registry 仍持有引用

  EXPECT_EQ(registry.size(), 1u);
  EXPECT_NEAR(registry.back()->cpu_mutable_data()[0], 99.0f, 1e-5f);
  EXPECT_NEAR(registry.back()->cpu_mutable_data()[1], 88.0f, 1e-5f);
}

/// 验证多次注册（多次 push_back）refcount 正确递增
TEST(ObjectPtrMigration, MultipleRegistrationsShareSameObject) {
  auto obj = make_object<Blob>(std::vector<int64_t>{3});
  obj->cpu_mutable_data()[0] = 77.0f;

  std::vector<ObjectPtr<Blob>> registry;
  registry.push_back(obj);  // refcount 2
  registry.push_back(obj);  // refcount 3
  registry.push_back(obj);  // refcount 4

  EXPECT_EQ(registry.size(), 3u);
  for (const auto& entry : registry) {
    EXPECT_EQ(entry.get(), obj.get());
    EXPECT_NEAR(entry->cpu_mutable_data()[0], 77.0f, 1e-5f);
  }

  // 通过任一 entry 写入，所有 entry 和原始 obj 都可见
  registry[1]->cpu_mutable_data()[0] = 123.0f;
  EXPECT_NEAR(obj->cpu_mutable_data()[0], 123.0f, 1e-5f);
  EXPECT_NEAR(registry[0]->cpu_mutable_data()[0], 123.0f, 1e-5f);
  EXPECT_NEAR(registry[2]->cpu_mutable_data()[0], 123.0f, 1e-5f);
}

/// 验证 registry 清空后原始对象仍有效（refcount 递减但未归零）
TEST(ObjectPtrMigration, RegistryClearPreservesOriginalObject) {
  auto obj = make_object<Blob>(std::vector<int64_t>{5});
  obj->cpu_mutable_data()[0] = 55.0f;

  {
    std::vector<ObjectPtr<Blob>> registry;
    registry.push_back(obj);  // refcount 2
    EXPECT_NEAR(registry.back()->cpu_mutable_data()[0], 55.0f, 1e-5f);
  }
  // registry 离开作用域，所有 entry 析构，refcount 降为 1

  EXPECT_NEAR(obj->cpu_mutable_data()[0], 55.0f, 1e-5f);
}

/// 验证移动语义：移动后原始指针为空，refcount 不增加
TEST(ObjectPtrMigration, MoveDoesNotIncreaseRefcount) {
  auto obj = make_object<Blob>(std::vector<int64_t>{2, 3});
  obj->cpu_mutable_data()[0] = 11.0f;
  auto* raw = obj.get();

  // 移动构造
  ObjectPtr<Blob> moved = std::move(obj);

  // 移动后：moved 指向原对象，原始变量为空
  EXPECT_EQ(moved.get(), raw);
  EXPECT_EQ(obj.get(), nullptr);

  // 数据通过 moved 可访问
  EXPECT_NEAR(moved->cpu_mutable_data()[0], 11.0f, 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// 场景三：FFI 边界 / 裸指针传递（WeightProcessor 模式）
// 覆盖：const ObjectPtr& 传参、lambda 适配、ObjectPtr 拷贝共享所有权
// ─────────────────────────────────────────────────────────────────────────────

/// 验证 ObjectPtr 拷贝构造保持指针相等与数据完整性
/// 对应迁移指南步骤 5：ObjectPtr 拷贝构造共享所有权，refcount 从裸指针恢复需通过 ObjectPtr 而非 GetRef
TEST(ObjectPtrMigration, CopyConstructorSharesPointerAndData) {
  auto obj = make_object<Blob>(std::vector<int64_t>{3, 3});
  obj->cpu_mutable_data()[0] = 33.0f;
  Blob* raw = obj.get();

  // ObjectPtr 拷贝构造 → refcount 递增，指向同一对象
  ObjectPtr<Blob> recovered = obj;

  // 恢复后的指针指向原对象
  EXPECT_EQ(recovered.get(), raw);
  EXPECT_EQ(recovered->cpu_data(), obj->cpu_data());
  EXPECT_NEAR(recovered->cpu_mutable_data()[0], 33.0f, 1e-5f);
}

/// 验证 ObjectPtr 拷贝后，原始对象析构不会导致悬空指针
/// 这是侵入式 refcount 相比 shared_ptr 的核心优势----
/// 注意：ObjectPtr 不可从裸指针直接构造（构造函数为 private），
/// 需通过拷贝已有 ObjectPtr 来共享所有权
TEST(ObjectPtrMigration, CopySurvivesSourceDestruction) {
  ObjectPtr<Blob> recovered;

  {
    auto obj = make_object<Blob>(std::vector<int64_t>{5});
    obj->cpu_mutable_data()[0] = 77.0f;
    obj->cpu_mutable_data()[1] = 66.0f;
    recovered = obj;  // 拷贝 → refcount 变为 2
  }
  // obj 析构，refcount 降为 1，recovered 仍持有引用

  EXPECT_NEAR(recovered->cpu_mutable_data()[0], 77.0f, 1e-5f);
  EXPECT_NEAR(recovered->cpu_mutable_data()[1], 66.0f, 1e-5f);
}

/// 验证 const ObjectPtr& 传参模式（模拟 FFI lambda 适配）
/// 对应迁移指南步骤 4：const ObjectPtr& 传递不增加 refcount
TEST(ObjectPtrMigration, ConstRefParameterDoesNotModifyOriginal) {
  auto obj = make_object<Blob>(std::vector<int64_t>{2, 2});
  obj->cpu_mutable_data()[0] = 55.0f;
  obj->cpu_mutable_data()[1] = 44.0f;

  // 模拟 FFI 注册：lambda 接收 const ObjectPtr&，内部 .get() 转裸指针
  auto read_and_sum = [](const ObjectPtr<Blob>& ref) -> float {
    // const ObjectPtr& → 不增加 refcount
    const Blob* b = ref.get();  // .get() 零开销
    return b->cpu_data()[0] + b->cpu_data()[1];
  };

  float result = read_and_sum(obj);  // 传递 const ObjectPtr&
  EXPECT_NEAR(result, 99.0f, 1e-5f);

  // 调用后原始对象不变
  EXPECT_NEAR(obj->cpu_mutable_data()[0], 55.0f, 1e-5f);
  EXPECT_NEAR(obj->cpu_mutable_data()[1], 44.0f, 1e-5f);
}

/// 验证 ObjectPtr 值传递 + 内部裸指针访问模式
/// 对应迁移指南步骤 4：需要共享所有权时按值传递 ObjectPtr
TEST(ObjectPtrMigration, ValuePassingForOwnershipTransfer) {
  auto obj = make_object<Blob>(std::vector<int64_t>{4});
  obj->cpu_mutable_data()[0] = 100.0f;

  // 模拟 WeightProcessor::ProcessWeight 模式：
  // 外部传 ObjectPtr（值传递），内部 .get() 转裸指针操作
  auto process = [](ObjectPtr<Blob> w) {
    // w 是值拷贝，refcount 已 +1
    // 内部用裸指针零开销访问
    Blob* b = w.get();
    b->cpu_mutable_data()[0] *= 2.0f;
    return b->cpu_mutable_data()[0];
  };

  float result = process(obj);  // 值传递 → refcount 临时 +1
  EXPECT_NEAR(result, 200.0f, 1e-5f);
  // 原始对象的值也被修改（因为共享同一对象）
  EXPECT_NEAR(obj->cpu_mutable_data()[0], 200.0f, 1e-5f);
}

/// 验证 vector<ObjectPtr<Blob>> 的批量操作
/// 对应 WeightRegistry 存储多个 Blob 的模式
TEST(ObjectPtrMigration, VectorOfObjectPtrsBulkOperations) {
  std::vector<ObjectPtr<Blob>> vec;

  // 场景：注册 5 个权重
  for (int i = 0; i < 5; ++i) {
    auto b = make_object<Blob>(std::vector<int64_t>{3});
    b->cpu_mutable_data()[0] = static_cast<float>(i * 10);
    b->cpu_mutable_data()[1] = static_cast<float>(i * 10 + 1);
    b->cpu_mutable_data()[2] = static_cast<float>(i * 10 + 2);
    vec.push_back(std::move(b));
  }

  EXPECT_EQ(vec.size(), 5u);

  // 验证所有元素独立
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(vec[i]->cpu_mutable_data()[0], static_cast<float>(i * 10), 1e-5f);
    EXPECT_NEAR(vec[i]->cpu_mutable_data()[1], static_cast<float>(i * 10 + 1), 1e-5f);
    EXPECT_NEAR(vec[i]->cpu_mutable_data()[2], static_cast<float>(i * 10 + 2), 1e-5f);
  }

  // 验证不同元素数据独立（修改一个不影响其他）
  vec[0]->cpu_mutable_data()[0] = 999.0f;
  EXPECT_NEAR(vec[0]->cpu_mutable_data()[0], 999.0f, 1e-5f);
  EXPECT_NEAR(vec[1]->cpu_mutable_data()[0], 10.0f, 1e-5f);  // 未受影响
  EXPECT_NEAR(vec[2]->cpu_mutable_data()[0], 20.0f, 1e-5f);  // 未受影响
}

/// 验证空的 ObjectPtr（nullptr 语义）
/// 对应迁移指南步骤 4：可能为空的可选参数用 const Foo*
TEST(ObjectPtrMigration, NullObjectPtr) {
  ObjectPtr<Blob> null_obj;
  // 默认构造的 ObjectPtr 为空
  EXPECT_EQ(null_obj.get(), nullptr);

  // 赋值后变为非空
  null_obj = make_object<Blob>(std::vector<int64_t>{1});
  EXPECT_NE(null_obj.get(), nullptr);
  EXPECT_NEAR(null_obj->cpu_mutable_data()[0], 0.0f, 1e-5f);
}

/// 验证 ObjectPtr 的 reset 行为
TEST(ObjectPtrMigration, ResetReleasesOwnership) {
  auto obj = make_object<Blob>(std::vector<int64_t>{2});
  obj->cpu_mutable_data()[0] = 88.0f;

  ObjectPtr<Blob> copy = obj;  // refcount 2
  EXPECT_EQ(copy.get(), obj.get());

  copy.reset();  // 释放引用，refcount 降为 1

  EXPECT_EQ(copy.get(), nullptr);
  EXPECT_NEAR(obj->cpu_mutable_data()[0], 88.0f, 1e-5f);  // 原始对象仍有效
}