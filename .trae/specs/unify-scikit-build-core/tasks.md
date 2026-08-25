# Tasks

## 任务总览

按「自下而上、先标准后应用」顺序执行：先确定 Canonical 风格与根目录改造基准，再逐子项目校准，最后文档与校验收尾。已用 scikit-build-core 的独立子项目（caffe-ffi、demo-ffi、npu-ffi、templates/native、native-ffi）相互独立，可并行处理。

## Task 1: 定义 canonical 风格与根目录枢纽包改造

将根目录 `pyproject.toml` 从 setuptools 切换为 scikit-build-core + 纯 Python 枢纽包，并将根 `CMakeLists.txt` 改为 `LANGUAGES NONE` + SKBUILD install(DIRECTORY) 模式，作为全仓库风格基准。

- [ ] SubTask 1.1: 改写根 `pyproject.toml`：`[build-system]` 改为 `scikit-build-core>=0.10` + `ninja>=1.11`，`build-backend = "scikit_build_core.build"`，移除 `wheel`；移除 `[tool.setuptools.packages.find]`；新增 `[tool.scikit-build]`（`cmake.build-type="Release"`、`ninja.make-fallback=false`、`build.verbose=true`）；保留 `[tool.pdm.workspace]` 与 `[dependency-groups]`。
- [ ] SubTask 1.2: 改写根 `CMakeLists.txt`：`project(xuanspace LANGUAGES NONE)`，移除 `add_subdirectory` 聚合，新增 `if(SKBUILD)` 注释说明枢纽包无包体（`install(DIRECTORY ...)` 留空占位）。
- [ ] 验证：`python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` 通过；根目录 `pip install -e ".[dev]"`（如环境允许）或干跑验证不依赖 setuptools。

## Task 2: 改造纯 Python 子项目（xuan-core）

将 `libs/xuan-core` 从 setuptools 切换为 scikit-build-core，并新增 CMakeLists.txt。

- [ ] SubTask 2.1: 改写 `libs/xuan-core/pyproject.toml`：切换 build-backend，移除 `[tool.setuptools.packages.find]` 与 `[tool.pdm.build]`，新增 `[tool.scikit-build]` `wheel.packages=["src/xuan_core"]`。
- [ ] SubTask 2.2: 新增 `libs/xuan-core/CMakeLists.txt`：`project(xuan_core LANGUAGES NONE)` + `if(SKBUILD) install(DIRECTORY src/xuan_core/ DESTINATION xuan_core) endif()`。
- [ ] 验证：`python -m build` 或 TOML 语法校验通过；无 setuptools 残留。

## Task 3: 更新 python 项目模板

将 `tools/templates/python/` 模板切换为 scikit-build-core，并新增 CMakeLists.txt。

- [ ] SubTask 3.1: 改写 `tools/templates/python/pyproject.toml` 为 canonical scikit-build-core 纯 Python 风格。
- [ ] SubTask 3.2: 新增 `tools/templates/python/CMakeLists.txt`（`LANGUAGES NONE` + `install(DIRECTORY src/{{package_name}}/ DESTINATION {{package_name}})`）。
- [ ] 验证：模板 `{{name}}`/`{{package_name}}` 变量占位完整，无 setuptools 引用。

## Task 4: 校准 tools 子项目（xs / okf）

统一 `tools/xs` 与 `tools/okf` 的 canonical 风格。

- [ ] SubTask 4.1: `tools/xs/pyproject.toml`：校准 `wheel.packages` 语义（当前 `["xs"]`，纯 Python 项目应指向包根 `["src/xs"]` 或显式 CMake install），统一键排序、补 `minimum-version="0.10"`、2 空格缩进。
- [ ] SubTask 4.2: `tools/okf/pyproject.toml`：统一键排序与缩进，补齐 `minimum-version`。
- [ ] SubTask 4.3: `tools/xs/CMakeLists.txt`、`tools/okf/CMakeLists.txt`：校验 `LANGUAGES NONE` 与 `install(DIRECTORY src/<pkg>/ DESTINATION <pkg>)` 一致。
- [ ] 验证：两个 pyproject 均含 scikit-build-core 与 ninja；`xs` 的 `wheel.packages` 与 CMake install 目标一致。

## Task 5: 校准原生/FFI 子项目（独立，可并行）

按 canonical 风格统一既有 scikit-build-core 子项目的 `minimum-version`、键排序、缩进，不改变构建语义（caffe-ffi、demo-ffi、npu-ffi、xuan-ext-demo 四个独立子项，SubTask 5.a–5.d 可并行）。

- [ ] SubTask 5.a: `libs/caffe-ffi/pyproject.toml` 格式校准（保留其平台 overrides 与 cmake.define，统一缩进与键序）。
- [ ] SubTask 5.b: `libs/demo-ffi/pyproject.toml` 格式校准。
- [ ] SubTask 5.c: `libs/npu-ffi/pyproject.toml` 格式校准。
- [ ] SubTask 5.d: `libs/xuan-ext-demo/pyproject.toml` 格式校准（补充 `minimum-version`、`wheel.packages` 语义、缩进）。
- [ ] SubTask 5.e: `tools/templates/native/pyproject.toml` 与 `tools/templates/native-ffi/pyproject.toml` 格式校准（对齐 `minimum-version`、缩进、注释风格）。
- [ ] 验证：每个文件保持 build backend 为 `scikit_build_core.build`，无 setuptools 残留，构建语义不变。

## Task 6: 文档更新

- [ ] `docs/build-system.md`：更新「pyproject.toml 标准」章节，scikit-build-core 为唯一后端，替换「纯 Python 项目配置」setuptools 示例为 scikit-build-core `LANGUAGES NONE` 示例。
- [ ] 验证：文档中的示例与 Task 1-3 实际产物一致，无 setuptools 描述残留。

## Task 7: 一致性校验工具

新增可复用的全仓库 pyproject.toml canonical 校验能力（放置于 `tools/` 或 `.agents/scripts/`），用于 CI 门禁。

- [x] SubTask 7.1: 编写校验脚本：遍历根目录、`libs/*`、`tools/*` 及 `tools/templates/**` 的 `pyproject.toml`，检查 `build-backend`、`requires` 含 scikit-build-core+ninja、无 `setuptools`/`tool.setuptools` 残留。
- [x] SubTask 7.2: 校验脚本纳入 `.github/workflows/ci.yml`（如仓库 CI 存在）或提供手动运行入口。
- [x] 验证：对当前仓库运行脚本，输出通过/失败清单，与本次改造结果一致。

## Task 8: 全仓库验证

按 checklist 逐项核查全仓库改造结果，确保所有 pyproject.toml 通过 canonical 风格校验、文档与模板一致。

- [x] SubTask 8.1: 运行 `check_pyproject_style.py` 全仓库扫描，11 个 pyproject.toml 全部通过。
- [x] SubTask 8.2: 核查全仓库 `build-backend` 均为 `scikit_build_core.build`，无 setuptools 残留（`grep -rnE "setuptools|tool\.setuptools|tool\.pdm\.build"`）。
- [x] SubTask 8.3: 核查根目录、xuan-core、python/native/native-ffi 模板的 CMakeLists.txt 均为 `LANGUAGES NONE` + `install(DIRECTORY ...)`。
- [x] SubTask 8.4: 核查 docs/build-system.md 将 scikit-build-core 描述为唯一后端，无 setuptools 描述残留。

## 任务依赖

- Task 1（根目录 + canonical 基准）是风格基准，Task 2/3/4 依赖 Task 1 确立的 canonical 约定。
- Task 5 各子项相互独立，可与 Task 2/3/4 并行。
- Task 6 依赖 Task 1-3 完成产物。
- Task 7 依赖 Task 1-5 全部完成。