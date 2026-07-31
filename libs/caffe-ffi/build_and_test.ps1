Set-Location "d:\spaces\SpecWeave\projects\xuanspace\libs\caffe-ffi"
cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ctest -R caffe_ffi_cpp_tests --output-on-failure -C Release
exit $LASTEXITCODE