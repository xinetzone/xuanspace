"""可逆效应（Disposable）抽象测试。"""

from okf.disposable import DisposableList, EffectMeta


def test_disposable_list_push_returns_undo():
    """DisposableList.push 返回可撤销该注册的函数。"""
    dl = DisposableList()
    calls: list[str] = []

    def effect() -> None:
        calls.append("effect")

    undo = dl.push(effect)
    assert len(dl) == 1

    # 撤销注册后，effect 不再被追踪
    undo()
    assert len(dl) == 0

    # 幂等：重复撤销无副作用
    undo()
    assert len(dl) == 0


def test_disposable_list_clear_returns_reverse_order():
    """DisposableList.clear 逆序返回（后注册先返回），并清空列表。"""
    dl = DisposableList()
    order: list[str] = []

    def first() -> None:
        order.append("first")

    def second() -> None:
        order.append("second")

    def third() -> None:
        order.append("third")

    dl.push(first)
    dl.push(second)
    dl.push(third)

    disposables = dl.clear()
    # 逆序返回
    assert disposables == [third, second, first]

    # 执行返回的逆函数后，观察执行顺序
    for dispose in disposables:
        dispose()
    assert order == ["third", "second", "first"]
    assert len(dl) == 0


def test_effect_meta_fields():
    """EffectMeta 包含 label 与 children 字段。"""
    child = EffectMeta(label="child")
    meta = EffectMeta(label="root", children=[child])
    assert meta.label == "root"
    assert meta.children == [child]
    # 默认 children 为空列表
    assert EffectMeta(label="solo").children == []
