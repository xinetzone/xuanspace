# 团队同步：tvm-ffi / caffe-ffi `pip install .` 卡点根因与治理

> 同步日期：2026-08-05
> 适用范围：caffe-ffi 开发、P0 环境验证、CI 构建
> 详细复盘见 [TVMFFI_CAFFEFFI_PIP_INSTALL_RETROSPECTIVE_20260805.md](TVMFFI_CAFFEFFI_PIP_INSTALL_RETROSPECTIVE_20260805.md)

---

## 一句话结论

反复卡在 `pip install .` 的**根因不是编译慢，而是"构建/运行时依赖来源不一致"**——构建隔离重复下载、setuptools-scm 依赖 git 元数据、Cython 扩展开关未开、vendored 与发布版本 double-source skew。

## 四大根因（现象 → 根因 → 对策）

| # | 现象 | 根因 | 对策 |
|---|------|------|------|
| 1 | `pip install .` 慢 / 离线失败 | 构建隔离每次重新下载构建依赖 | `pip install --no-build-isolation -e .`，CI 先显式装 `scikit-build-core ninja cmake cython setuptools-scm` |
| 2 | 换环境报 `setuptools-scm unable to detect version` | 版本号从 git 元数据生成，脱离 `.git`（Docker 复制）即失效 | 注入 `SETUPTOOLS_SCM_PRETEND_VERSION=<version>`；或 Docker COPY 保留 `.git` |
| 3 | 运行时 `ModuleNotFoundError: tvm_ffi.core` | `TVM_FFI_BUILD_PYTHON_MODULE` 未开启，Cython core 扩展未编译 | 构建时显式 `-DTVM_FFI_BUILD_PYTHON_MODULE=ON` |
| 4 | Windows `WinError 127` | 头文件用 vendored（新 API），运行时 DLL 来自 site-packages（旧 API），版本 skew | 统一两处版本；Windows 需 vendored DLL；P0 Linux 环境做运行时冒烟 |

## 关键反常识

- **构建通过 ≠ 运行通过**：`pip install` 成功只代表编译链接成功，运行时仍需重新解析动态库，后者的版本来源常被忽略（`WinError 127` 因此漏过）。
- **"卡住"的等待多在隔离环境重复装依赖**，而非真正的 C++ 编译。

## 治理模式：构建与运行依赖一致性治理（build-runtime parity）

任何基于 `scikit-build-core` + setuptools-scm + Cython 扩展的项目，多环境安装反复失败时，按序执行：

1. 关闭构建隔离：`pip install --no-build-isolation -e .` + 显式装构建依赖
2. 注入动态版本：`SETUPTOOLS_SCM_PRETEND_VERSION=<version>`
3. 显式开启 Cython 扩展开关（如 `TVM_FFI_BUILD_PYTHON_MODULE=ON`）
4. 校验 `requires-python` 与目标环境匹配
5. 构建后运行时冒烟（`import` + 动态库 `ldd`/`dll` 解析）
6. 统一 vendored 与发布依赖版本，避免 double-source skew
7. 固化为脚本/CI step，避免每次手工重试

**反模式**（勿犯）：❌ 不加 `--no-build-isolation` ❌ 忽略 git 版本依赖 ❌ 只验编译不验运行时 ❌ 头文件与运行时用两套版本。

## 本次已交付

| 交付物 | 位置 | 用途 |
|--------|------|------|
| P2 数据 I/O 算子单元测试 | `tests/python/test_p2_data_io_ops.py` | Data/ImageData/HDF5Data 前向 + 回调填充（12 用例，P0 环境全部通过） |
| P0 定时任务脚本 | `scripts/p0_scheduled_test.sh` | flock 防并发 + 时间戳日志 + cron/systemd 接入 |
| TVM-FFI 依赖检查脚本 | `scripts/ci_check_tvmffi.py` | 跨平台 core 扩展 / 动态链接 / FFI 冒烟核验 |
| CI 集成 | `.github/workflows/ci.yml` | build-and-test 作业新增 "TVM-FFI dependency loading check" step |

## 后续待办

- [ ] C-4：将「`--no-build-isolation` + 版本注入 + 运行时冒烟」固化为标准构建脚本，验证干净环境一次通过
- [ ] 为 vendored tvm-ffi 补充 Windows DLL 构建，根治 `WinError 127`

## 参考

- 完整复盘：[TVMFFI_CAFFEFFI_PIP_INSTALL_RETROSPECTIVE_20260805.md](TVMFFI_CAFFEFFI_PIP_INSTALL_RETROSPECTIVE_20260805.md)