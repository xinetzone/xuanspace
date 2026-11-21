# xuan-fea

OpenType Feature（`.fea`）AST 构建原语，萃取自 [Maple Mono](https://github.com/subframe7536/maple-font) v7.9
的 `source/py/feature/ast.py`，面向「用 Python 对象描述特征文件、由渲染器统一处理缩进与空行」这一类需求。

## 功能

| 模块 | 公开 API | 作用 |
|------|----------|------|
| `xuan_fea.dsl` | `Line`、`Clazz`、`Lookup`、`Feature`、`FeatureWithDocs` | AST 节点：叶子行、字形类、查找块、特征块、带文档特征 |
| `xuan_fea.dsl` | `CharacterVariant`、`StylisticSet` | `cv01`–`cv99` / `ss01`–`ss20` 及自动生成的标签名块 |
| `xuan_fea.dsl` | `gly`、`gly_seq`、`cls`、`cls_states` | 字形名规范化（标点映射、`.liga` 补全）、内联类、类声明 |
| `xuan_fea.dsl` | `subst`、`subst_map`、`subst_liga`、`ign` | 单条替换、批量替换、连字查找块、排除规则 |
| `xuan_fea.dsl` | `langsys`、`lang`、`script` | `languagesystem` / `language` / `script` 声明 |
| `xuan_fea.dsl` | `create`、`flatten_to_lines`、`recursive_iterate` | 渲染与树遍历：折叠空行、段落间补空行、统一缩进 |
| `xuan_fea.dsl` | `EMPTY_FEAT_CONTENT`、`EMPTY_FEAT_SYMBOL`、`clone_empty`、`filter_empty` | 「关闭某特征」时使用的占位内容与过滤 |

本库**零第三方依赖**：不引入 `fonttools`，只负责生成 `.fea` 文本；编译进字体是调用方的事。
边界因此很清晰——`xuan-fea` 可以独立测试，也可以被任意字体工程复用。

## 安装

```bash
pip install -e ".[test]"
```

## 快速使用

```python
from xuan_fea import CharacterVariant, Clazz, Feature, Line, create, subst

digit = Clazz("Digit", ["zero", "one", "two"])

cv = CharacterVariant(
    id=1,
    desc="Slashed `zero`",
    content=[subst(None, "zero", None, "zero.slash")],
    version="7.9",
    example="0",
)

calt = Feature("calt", [Line("sub one two by one_two.liga;")], "7.9")

print(create([digit, cv, calt]))
```

连字查找块由 `subst_liga` 一次生成，包括上下文规则与排除规则：

```python
from xuan_fea import create, subst_liga

lookup = subst_liga("!=", ign_prefix="!", ign_suffix="?")
print(create([lookup]))
```

## 与上游的关系

- 提取范围：仅 `source/py/feature/ast.py`，不包含构建流水线、字体合并与特征冻结。
- 行为基线：`tests/test_golden_upstream.py` 用同一棵节点树分别驱动上游实现与本库，
  对 `create()` 结果及全部辅助函数输出做逐字节比对；上游路径不存在时该用例自动跳过。
- 有意偏差：可变默认参数改为不可变默认值；`clone_empty` 的异常消息修正拼写。两者均不影响输出。

## 来源与许可

代码萃取自 Maple Mono（作者 subframe7536），上游以 SIL Open Font License 1.1 发布
（见 `notebook/libs/maple-font/OFL.txt`）。本仓 `pyproject.toml` 沿用 Apache-2.0 登记，
对外分发前请复核两许可证的兼容性，并保留上游署名。

## 测试

```bash
python -m pytest
```
