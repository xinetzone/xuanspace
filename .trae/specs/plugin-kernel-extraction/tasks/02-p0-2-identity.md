> 分卷来源：`tasks.md` 原文 L362–L411（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：01-plan-p0-1.md · 下一卷：03-p0-3-disposal.md
> 主题：增量一 P0-2：epoch 摘要身份
# 增量一：P0-2（epoch 摘要身份）— 执行记录

> 2026-11-21 追加。自工作区级设计文档 `xuanspace-pluginization-design.md` §7 P0-2 派生。
> 本增量**改变运行语义**（epoch 身份由对象地址改为内容摘要），与 P0-1 的「零行为变更」
> 不变量不同，故独立成节，不回写 Task 1–8。

## Task P2-1: epoch 改用摘要身份并纳入注入配置

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 8（P0-1 提交 `4a3dc76`）
- **Description**:
  - `plugin.py` 新增 `_digest(value)`：标量取「类型名 + 字面量」；具备 `__qualname__` 者取
    `module:qualname:sha256(source)[:16]`；源码不可得（内建对象、动态生成实例）回退
    `sha256(repr(value))[:16]`；
  - 新增 `_config_digest(config)`：`json.dumps(sort_keys=True, ensure_ascii=False, default=repr)`
    后取 `sha256[:16]`，空字典或 `None` 返回 `"none"`；
  - `_compute_epoch()` 的 `id(impl)` 替换为
    `f"{spec.name}:{_digest(impl)}:{_config_digest(spec.config)}"`，依赖缺失仍返回 `_INACTIVE`；
  - 新库 `pyproject.toml` 补 `[project.optional-dependencies] test = ["pytest>=8"]` 与
    `[tool.pytest.ini_options] testpaths = ["tests"]`，使 CI 可进入库目录直接执行；
  - `tools/okf` 两处 epoch 断言同步更新。
- **Acceptance Criteria Addressed**: 装配结果跨进程可复现；配置键序不影响 epoch
- **Test Requirements**:
  - `rule` TR-P2.1: 同一实现两次摘要相同；证据：`test_digest.py` 确定性用例
  - `rule` TR-P2.2: 摘要文本不含对象地址；证据：断言不含 `0x` 前缀与 `id` 数值
  - `rule` TR-P2.3: 同类型不同实例可区分；证据：双 `object()` 实例用例
  - `rule` TR-P2.4: 配置键序无关、`None` 与空字典等价；证据：`_config_digest` 用例
  - `rule` TR-P2.5: 跨进程一致；证据：`subprocess` 两次子进程输出比对
  - `rule` TR-P2.6: `tools/okf` 通过数不回归；证据：pytest 摘要行
- **Completion Evidence**（2026-11-21 实测）:
  - 提交：`a39c72d`（完整 `a39c72d903138699a9f43c9a7f0542e9a9fbe2b1`），
    信息 `feat(plugin): epoch 改用摘要身份并纳入注入配置`，6 文件 +183 / −10。
  - 新增 `libs/xuan-plugin/tests/test_digest.py`（109 行、11 项）：摘要确定性、源码不同则摘要不同、
    不含 `0x`/id 地址、实例可区分、标量摘要、配置键序无关、`None` 与空等价、配置值变化可测、
    epoch 含 config、epoch 组成相符、`test_epoch_reproducible_across_processes`
    （以 `[sys.executable, "-c", code]` 起两次子进程、`cwd=TESTS_DIR` 比对）。
  - TR-P2.1～P2.5 通过：新库 11 passed。
  - TR-P2.6 通过：`cd tools/okf && python -m pytest -q` → **259 passed**，与 P0-1 基线一致。
  - okf 侧断言同步：`test_plugin.py` 与 `test_plugin_extra.py` 的期望改为
    `f"dep:{_digest(impl)}:{_config_digest(None)}"`。
  - CI 同步：quality job 改为 `pip install -e "libs/xuan-plugin[test]"`，并在 okf 之前插入
    `cd libs/xuan-plugin && python -m pytest`；test job 的 mypy 范围追加 `libs/xuan-plugin`。
  - `mypy` 期间修正 1 处：`inspect.getsource` 形参为联合类型，动态取值无法静态收窄，
    显式标注 `target: Any`。
- **Notes**: 边界已写入 `_digest` 的 docstring——**无源码实例回退 `repr` 摘要**，
  同一进程内可区分，但不保证跨进程稳定。这是本方案对「内建对象作依赖实现」场景的已知上限。

---

