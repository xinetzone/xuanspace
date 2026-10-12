# xuan-compose 功能完整性独立验证报告（七概念 F→V→I）

> 验证日期：2026-09-16（T1-T12 交付并推送后的**独立复验**，不复用交付期自审结论）
> 上游基线：containers/podman-compose v1.6.0，pinned commit
> `e3df10472e194ab6d547b5ad25542c5c79e1a5fb`（GPL-2.0-only）
> 方法链：F（第一性原理重建"功能全集"标尺）→ V（depth=deep，5 攻击者四路并行证伪，
> 含 1 轮回归对抗）→ I（差异四元组定级）；纯只读审计 + 临时目录构建实证，未改源码逻辑
> 关联文档：[parity 入口](../parity/index.md)、
> [T11 对抗审查](../review/2026-09-16-t11-adversarial-review.md)、
> [交付总结报告](../delivery/2026-09-16-delivery-summary.md)
>
> 历史说明：2026-10-10 原子化重写迁自
> `docs/2026-09-16-xuan-compose-full-parity-verification.md`，数字与结论零增删改
> （标题与导航性文字随原子化调整；§4.1 一处指向 `xuanspace/CMakeLists.txt`
> 的链接因该文件后续被提交 `f293ed6` 删除而改为纯文本）；
> 原 §3/§5 清单已合并至[问题处置台账](../findings-ledger.md)（§E/§C/§D）。

---

## 1. 总结论

**功能已全部迁移，证伪失败：0 个 P0（功能丢失/错误）。**

"940 测试全绿 + 142 公开符号 100% 映射"只覆盖功能全集的一个维度。本次从第一性原理
重建 7 维标尺，经机械 AST 对拍、3 个 fresh-context 对抗代理与运行时实证，未发现任何
函数/方法/分支/字面量/分派链/动态机制的静默丢失；[upstream-quirks.md](../parity/upstream-quirks.md)
登记的全部上游怪癖逐一在位。

审计中发现的唯一实质问题（**分发面 P1：纯 Python 包产出平台钉死 wheel**）已在本次
验证中**修复并实证**（见第 4 节）；其余为登记补记类 P2（[findings-ledger §C](../findings-ledger.md#c-p2)）；
两个环境层面 P1 见 [findings-ledger §E](../findings-ledger.md#e-env)。

## 2. 七维证据链

| # | 维度 | 攻击方式 | 结论 |
|---|---|---|---|
| ① | 公开符号名称 | AST 盘点复核 | 142/142 映射；3 项删除/重命名全部已登记（D1/D2/D6） |
| ② | **全量顶层函数（含 `_` 私有）** | 上游 126 个 vs 新库跨模块并集机械对拍 | **0 缺失**；唯一缺名 `wrapped` 为 cmd_run/cmd_parse 装饰器闭包，随 D1/D2 删除 |
| ③ | **类方法全集（含私有）** | 逐类逐方法对拍 | Podman 9/9、数据类方法全等；ComposeEngine 少的 3 个（run/_parse_args/_init_global_parser）即 D2/D3/D10 迁入 CLI 层；4 个 token 类系 `var_interpolate` 嵌套类（与上游同构）；6 处空体全为 `@overload`/抽象方法存根，**无桩实现** |
| ④ | CLI 分派闭环 | 不信快照，解开 vendor 装饰器闭包做**运行时活体**对拍 | 24/24 命令集合/parser 挂载顺序/handler 配对零差集；help/version/未知命令/无参 4 归宿与退出码（含 -1、2）一致；9 个退出点 1:1；24 handler 全部 async 且无空体 |
| ⑤ | 常量/正则/字面量 | 800 个字符串字面量机器差分 + 人工复核 | 6 个正则逐字一致；label 键 28 位点、环境变量 11:11、systemd 模板逐字一致；未解释差异 0（15 个机器告警逐一证伪：9 个 ESC 转义假象、5 个 f-string 合并假象、1 个 D1 已登记删除） |
| ⑥ | 函数体忠实度 + 测试断言强度 | 新库 197 个函数节点 AST 归一化全量对拍 + 30 个点名函数人工复核 + 假绿排查 | 0 未登记逻辑改写；**0 skip / 0 xfail / 0 空心自测 mock**；断言落在真实 podman 命令行/退出码/stdout；kill 重复分支、ls `except Exception: break`、logs 空 `max()` ValueError、PullPolicyAction 静默 return 等怪癖全等保留 |
| ⑦ | 动态/入口/打包面 | signal/asyncio/子进程位点对拍 + 真实 wheel/sdist 构建 + `-m`/console script 实证 | 运行时 **0 P0**（signal 监督、4 个 create_subprocess_exec、os.execlp、tempfile、asyncio 拓扑 18:18 位点全在）；打包面发现 1 个 P1（第 4 节已修复） |

### 关键运行时实证（非静态推断）

- `python -m xuan_compose version --short` 退出 0；`xuan-compose` console script 可导入；
  bare `import xuan_compose` 零副作用，`!override/!reset` 仅在 CLI 链路导入 merge 后注册
  （全库唯一 `yaml.safe_load` 与标签类同模块，不存在真实加载路径漏注册）。
- argparse 无参/help 实测退出码 **-1**，未知命令退出码 2，`-v`/`--version up` 均路由 version。
- 注入返回码 42 的 handler 实测透传为 `SystemExit(42)`。
- 真实 YAML 入口下 `depends_on: !override` 的 list/dict 两形态上下游对拍（详见 [P2-1](../findings-ledger.md#ledger-p2-1)）。

## 3. 环境层面两个 P1（非产品代码缺陷，影响验证可复现性）

已合并至 [findings-ledger §E](../findings-ledger.md#e-env)：①陈旧 editable 快照劫持导入；
②原生 Windows 34 failed（POSIX 路径假设差异，权威门禁在 WSL）。

## 4. 已修复：分发面 P1（纯 Python 包产出平台钉死 wheel）

<a id="section-4"></a>

### 4.1 问题

仓库根存在脚手架模板生成的 `CMakeLists.txt`（`LANGUAGES NONE`，仅 1 行 install 目录），
顶层超工程 `xuanspace/CMakeLists.txt` 明确**不 add_subdirectory
聚合任何子项目**，该文件无任何外部引用；但 scikit-build-core 探测到它即进入 CMake 构建：

- 产出 `xuan_compose-0.1.0-cp314-cp314-win_amd64.whl`（`Root-Is-Purelib: false`），
  Linux/macOS 用户无法复用；
- sdist 本地构建需要 cmake≥3.26 + ninja，但二者未声明在 build-system requires，干净环境
  `pip install` 必失败（本机成功仅因全局预装 cmake 4.2/ninja 1.13）。

### 4.2 修复（本次变更，2 个文件）

- **删除** `CMakeLists.txt`（无外部引用的脚手架残留）；
- `pyproject.toml`：删除 `cmake.build-type`/`ninja.make-fallback` 两个 CMake 专用键，
  新增 `[tool.scikit-build.wheel] cmake = false` + `packages = ["src/xuan_compose"]`。
  该布尔是 scikit-build-core 1.0.3 决定是否调用 CMake 的真实开关
  （`scikit_build_core/build/wheel.py:340 if settings.wheel.cmake`；同时决定
  purelib/platlib 标签，`common_wheel_helpers.py:95`）。
  注：曾试验 `cmake.version = ""`，实证**不能**绕过 CMake 配置，已弃用。

### 4.3 修复后实证（临时目录干净 PEP 517 构建，仓库零构建残留）

| 验证项 | 结果 |
|---|---|
| wheel 标签 | `xuan_compose-0.1.0-py3-none-any.whl`，`Root-Is-Purelib: true`，97651 字节 |
| 内容完整性 | 39 个 .py（cli 3 / commands 10 / translate 9 全收录）+ py.typed + GPL-2.0 LICENSE |
| console_scripts | `xuan-compose = xuan_compose.cli.main:main` 正确生成 |
| 构建过程 | 无 CMake configure/build、无 ninja 参与 |
| sdist | `xuan_compose-0.1.0.tar.gz`（271449 字节），含 pyproject + 39 .py、不含 CMakeLists |
| sdist → wheel | 从 tar 包重建成功，产物仍为 `py3-none-any` 且与直构 wheel 字节级同尺寸 |
| 回归门禁 | WSL `PYTHONPATH=src pytest`：**940 passed / 74 subtests / 覆盖率 92%**，与修复前基线一致 |

遗留无害项：sdist 阶段 scikit-build-core 1.0.3 会打印一次
"CMakeLists.txt not found ... Using 3.15 as a fall-back" 警告（仅元数据探测，不调用
CMake、不影响构建）；后续升级后端版本可观察是否消失。

> 同源缺陷提示：脚手架模板 `tools/templates/python/CMakeLists.txt` 仍会给未来纯 Python
> 子项目生成同类配置，建议在模板层同步移除 CMakeLists 并改用 `wheel.cmake=false`
> （属 tools 区域变更，本次未触碰，留待决策；另见 [findings-ledger §D](../findings-ledger.md#d-p1-fixed)）。

## 5. P2 遗留清单

10 项登记/加固类遗留（非功能丢失，均未改代码）的完整证据与定级理由见
[findings-ledger §C](../findings-ledger.md#c-p2)：P2-1 normalize_service assert 分歧、
P2-2 COMPOSE_DEFAULT_LS 双份副本、P2-3 D9 登记不完整、P2-4 版本号双轨、
P2-5 logger 名与外观文本、P2-6 systemd 共享命名空间、P2-7 bash completion 未携带、
P2-8 updown 深层编排无单测、P2-9 轮询断言偏弱、P2-10 Python <3.14 误用晦涩。

## 6. 三条洞察（I 阶段）

1. **"名称全集"给的是虚假安全感。** 142 公开符号映射早已 100%，真正的完备性只有在
   含私有的 126 函数、逐类方法、800 字面量、24 条活体分派链全部对拍后才成立。翻译式
   重构的验收标准应把"私有顶层符号零缺失 + handler 活体分派 + 字面量差分"固化为标准栏位。
2. **静态可疑 ≠ 行为分歧，可达性决定定级。** 本次最强攻击（P2-1 的 assert 分歧）在真实
   YAML 入口不可达；差异分级必须先回答"用户走哪条公开路径能观察到它"，再谈严重性。
3. **运行时对等 ≠ 分发可用。** 功能 0 P0 不代表用户能装上——纯 Python 包被脚手架拖出
   平台钉死 wheel 是独立于代码正确性的发布缺陷，需构建实证才能发现，代码审查不可见。

## 7. 复现命令

```bash
# WSL 权威门禁（POSIX，零真实 podman）
cd /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/xuan-compose
PYTHONPATH=src ~/.venvs/xuan-gate/bin/pytest -q --cov=src/xuan_compose --cov-report=term-missing

# 纯 Python wheel / sdist 干净构建（临时目录，无需 cmake/ninja）
python -m pip wheel <repo> --no-deps --no-build-isolation -w dist/
python -m build --sdist --no-isolation
# 期望产物：xuan_compose-0.1.0-py3-none-any.whl（Root-Is-Purelib: true）
```
