"""事件分发兼容模块。

五种分发模式（``on`` / ``emit`` / ``bail`` / ``parallel`` / ``serial`` /
``waterfall``）已作为正式方法移入 :class:`okf.context.Context`，原先对
``Context`` 的 monkey-patch 已移除。保留本模块仅为向后兼容，使
``from okf import events`` 依旧可导入而不产生副作用。
"""
