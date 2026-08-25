# Checklist

## 说明
本清单用于系统验证「统一玄境 scikit-build-core 构建风格」改造是否满足 spec 要求。逐项核查可作为可复现证据。
验证命令示例：
- 「构建后端统一」：`grep -r "build-backend" --include=pyproject.toml .`（应全部为 `scikit_build_core.build`）
- 「无 setuptools 残留」：`grep -rnE "setuptools|tool\.setuptools" --include=pyproject.toml .`（应无命中，模板占位除外）
- 「TOML 语法」：`python -c "import tomllib,glob;[tomllib.load(open(f,'rb')) for f in glob.glob('**/pyproject.toml',recursive=True)]"`

## 根目录枢纽包改造
- [x] 根 `pyproject.toml` 的 `build-backend` 为 `scikit_build_core.build`，`requires` 含 `scikit-build-core>=0.10` 与 `ninja`
- [x] 根 `pyproject.toml` 无 `[tool.setuptools.packages.find]` 残留
- [x] 根 `pyproject.toml` 保留 `[tool.pdm.workspace]` 与 `[dependency-groups]`
- [x] 根 `CMakeLists.txt` 声明 `LANGUAGES NONE`，无 `add_subdirectory` 聚合原生扩展
- [x] 根 TOML 语法通过 `tomllib` 解析

## 纯 Python 子项目 / 模板
- [x] `libs/xuan-core/pyproject.toml` 使用 scikit-build-core，无 setuptools/`tool.pdm.build` 残留
- [x] `libs/xuan-core/CMakeLists.txt` 存在且为 `LANGUAGES NONE` + `install(DIRECTORY src/xuan_core ...)`
- [x] `tools/templates/python/pyproject.toml` 使用 scikit-build-core，模板占位完整
- [x] `tools/templates/python/CMakeLists.txt` 存在且为 `LANGUAGES NONE` + `install(DIRECTORY src/{{package_name}} ...)`

## tools 子项目校准
- [x] `tools/xs/pyproject.toml` `wheel.packages` 语义与 CMake install 目标一致，含 `minimum-version="0.10"`
- [x] `tools/okf/pyproject.toml` 键排序/缩进统一，含 `minimum-version="0.10"`
- [x] `tools/xs/CMakeLists.txt` 与 `tools/okf/CMakeLists.txt` 为 `LANGUAGES NONE` + 一致 `install(DIRECTORY src/<pkg> ...)`

## 原生/FFI 子项目校准（构建语义不变）
- [x] `libs/caffe-ffi/pyproject.toml` build-backend 正确，保留平台 overrides，格式统一
- [x] `libs/demo-ffi/pyproject.toml` build-backend 正确，格式统一
- [x] `libs/npu-ffi/pyproject.toml` build-backend 正确，格式统一
- [x] `libs/xuan-ext-demo/pyproject.toml` 含 `minimum-version`、`wheel.packages` 语义正确
- [x] `tools/templates/native/pyproject.toml` 与 `tools/templates/native-ffi/pyproject.toml` 格式统一（native 模板补 `ninja`）
- [x] 以上原生/FFI 项目均无 setuptools 残留

## 全仓库一致性
- [x] 全仓库扫描：所有 `pyproject.toml` 的 `build-backend` 均为 `scikit_build_core.build`（11 个全通过）
- [x] 全仓库扫描：无 `tool.setuptools.packages.find` / 纯 setuptools build-backend 残留
- [x] 全仓库扫描：所有 `pyproject.toml` 均可通过 `tomllib` 解析（11 个全通过）

## 文档
- [x] `docs/build-system.md` 将 scikit-build-core 描述为唯一构建后端，替换/弃用 setuptools 示例（原生 C++ 示例已与仓库实际产物对齐）

## 校验工具
- [x] 一致性校验脚本可运行，能识别符合/不符合 canonical 风格的文件
- [x] 校验脚本（或运行入口）已纳入仓库（`scripts/check_pyproject_style.py` + CI quality 任务步骤）
- [x] 校验脚本对当前改造后仓库输出「全通过」或无新增失败（11 个全通过）