"""okf CLI 入口模块，允许通过 python -m okf 运行。"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - 进程入口，由集成测试覆盖
    main()  # pragma: no cover
