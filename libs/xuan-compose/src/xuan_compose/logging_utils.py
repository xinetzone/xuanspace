# SPDX-License-Identifier: GPL-2.0-only
"""日志装配。

上游 1.6.0 只有两处日志配置：模块级 ``log = logging.getLogger(__name__)``
与 main 中的 ``logging.basicConfig(level=("DEBUG" if verbose else "WARN"))``
（第 109、3170 行）。本模块集中提供二者，包导入不配置 root logger、不产生输出。
"""

import logging

LOGGER_NAME = "xuan_compose"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


#: 各层共用的模块级 logger（语义对齐上游 ``log``）。
log = get_logger()


def configure_logging(verbose: bool) -> None:
    """对齐上游 main：仅按 --verbose 切换 DEBUG/WARN 级别。"""
    logging.basicConfig(level=("DEBUG" if verbose else "WARN"))
