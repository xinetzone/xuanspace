"""OKF Bundle 加载器：遍历目录树，加载概念与保留文件。"""

from pathlib import Path

from .frontmatter import parse_concept
from .models import Bundle, Concept


def load_bundle(root: Path) -> Bundle:
    """遍历 Bundle 目录树，将所有 ``.md`` 文件分类加载。

    - ``index.md`` → ``Bundle.indices``
    - ``log.md`` → ``Bundle.logs``
    - 其他 ``.md`` → 解析为 :class:`Concept`，以相对路径（去 ``.md`` 后缀）作为概念 ID
    """
    concepts: dict[str, Concept] = {}
    indices: list[Path] = []
    logs: list[Path] = []

    for md_file in root.rglob("*.md"):
        if md_file.name == "index.md":
            indices.append(md_file)
        elif md_file.name == "log.md":
            logs.append(md_file)
        else:
            concept = parse_concept(md_file)
            relative = md_file.relative_to(root)
            concept_id = str(relative.with_suffix("")).replace("\\", "/")
            concepts[concept_id] = concept

    return Bundle(root=root, concepts=concepts, indices=indices, logs=logs)


def load_concept(bundle: Bundle, concept_id: str) -> Concept | None:
    """根据概念 ID 从 Bundle 中查找 Concept，不存在时返回 ``None``。"""
    return bundle.concepts.get(concept_id)
