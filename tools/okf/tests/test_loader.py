"""``okf.loader`` Bundle 加载器的单元测试。"""

from __future__ import annotations

from pathlib import Path

from okf.loader import load_bundle, load_concept
from okf.models import Bundle, Concept


class TestLoadBundle:
    def test_loads_sample_bundle(self, sample_bundle_path: Path) -> None:
        bundle = load_bundle(sample_bundle_path)
        assert isinstance(bundle, Bundle)
        assert bundle.root == sample_bundle_path

    def test_classifies_index_and_log(self, sample_bundle_path: Path) -> None:
        bundle = load_bundle(sample_bundle_path)
        assert len(bundle.indices) == 1
        assert len(bundle.logs) == 1
        assert bundle.indices[0].name == "index.md"
        assert bundle.logs[0].name == "log.md"
        # index.md / log.md 应归入 indices/logs，而不进入 concepts
        assert "index" not in bundle.concepts
        assert "log" not in bundle.concepts

    def test_concept_ids(self, sample_bundle_path: Path) -> None:
        bundle = load_bundle(sample_bundle_path)
        assert set(bundle.concepts) == {
            "concepts/metrics/revenue",
            "concepts/metrics/active_users",
            "concepts/tables/customers",
        }

    def test_recursive_load(self, sample_bundle_path: Path) -> None:
        bundle = load_bundle(sample_bundle_path)
        # concepts/metrics 与 concepts/tables 两个子目录均被递归加载
        assert len(bundle.concepts) == 3

    def test_concept_type_extracted(self, sample_bundle_path: Path) -> None:
        bundle = load_bundle(sample_bundle_path)
        assert bundle.concepts["concepts/metrics/revenue"].type == "Metric"
        assert bundle.concepts["concepts/tables/customers"].type == "Table"


class TestLoadConcept:
    def test_existing_concept(self, sample_bundle_path: Path) -> None:
        bundle = load_bundle(sample_bundle_path)
        concept = load_concept(bundle, "concepts/metrics/revenue")
        assert isinstance(concept, Concept)
        assert concept.title == "Revenue"

    def test_missing_concept_returns_none(self, sample_bundle_path: Path) -> None:
        bundle = load_bundle(sample_bundle_path)
        assert load_concept(bundle, "nonexistent/id") is None


class TestLoadBundleTmpPath:
    def test_tmp_path_bundle(self, tmp_path: Path) -> None:
        (tmp_path / "index.md").write_text("# Index\n", encoding="utf-8")
        (tmp_path / "log.md").write_text("# Log\n", encoding="utf-8")
        concept_dir = tmp_path / "concepts"
        concept_dir.mkdir()
        (concept_dir / "foo.md").write_text(
            "---\ntype: Metric\ntitle: Foo\n---\n# Foo\n", encoding="utf-8"
        )

        bundle = load_bundle(tmp_path)
        assert len(bundle.indices) == 1
        assert len(bundle.logs) == 1
        assert set(bundle.concepts) == {"concepts/foo"}
        assert bundle.concepts["concepts/foo"].title == "Foo"
