# Checklist — slim-mystx-presets

## 声明层

- [x] `ConfigSpec` 的默认值字段同时接受字面值与 `Callable[[_BuildContext], Any]`，装配时可调用值被求值
- [x] `_constant()` 已删除，全仓无引用
- [x] `SITE_CONFIG_SPEC` 每条声明占一行，声明块 ≤35 行
- [x] `SITE_CONFIG_SPEC` 的键序与改动前完全一致（按显式键列表断言）
- [x] 19 个键的取值与改动前快照逐键相等
- [x] `_BuildContext.gettext_compact` 注解为 `bool | str`
- [x] 可变默认值（list/dict）全部经工厂逐次拷贝，无字面量直传

## 校验时机

- [x] 键集一致性校验在模块导入期执行一次
- [x] `apply_config()` 不再执行 `undeclared` 差集检查，且无残留死代码
- [x] `build_config()` 单次调用不触发键集比较
- [x] 导入期校验失败时抛出的 `RuntimeError` 含缺失键与多余键名

## 极简预设

- [x] `minimal_myst.build_config` 中不存在与 `params` 同值的重复写入
- [x] 返回值仍为含 `setup` 的扁平字典，`setup` 可调用且执行顺序为「注册钩子 → 用户 setup」

## 站点画像

- [x] `mystx_docs.configure` 以整体深拷贝注入所有字典/列表
- [x] 同进程两次装配得到的可变对象互不相同
- [x] `doc/conf.py` 未修改

## 公开契约

- [x] `presets.__all__` 仍为 4 项，`SiteConfig` 仍仅在 `TYPE_CHECKING` 下导出
- [x] `build_minimal_myst_config` / `build_mystx_site_config` / `apply_site_config` / `configure_site_fonts` 签名不变
- [x] 未新增公共 API，未新增运行期依赖

## 质量门

- [x] `pytest tests/unit` ≥134 passed / 3 skipped
- [x] `ruff check src/mystx/presets` 零告警
- [x] `sphinx-build -b html doc <临时目录>` 退出码 0，告警数不高于基线
- [x] `presets` 五个模块总行数 ≤620
- [x] 每项任务一个原子提交，提交信息符合 Conventional Commits 且主体为中文
