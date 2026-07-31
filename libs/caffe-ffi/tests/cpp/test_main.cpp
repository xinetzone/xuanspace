#include "test_harness.hpp"

int main(int argc, char* argv[]) {
  // Optional first argument: test filter (substring match on suite name or full test name)
  const char* filter = (argc > 1) ? argv[1] : nullptr;
  return caffe_ffi::testing::RunAllTests(filter);
}
