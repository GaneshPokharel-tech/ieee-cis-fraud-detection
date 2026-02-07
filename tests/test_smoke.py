from pathlib import Path

def test_repo_structure_exists():
    assert Path("README.md").exists()
    assert Path("src").exists()
    assert Path("configs/baseline.yaml").exists()
    assert Path("reports").exists()

def test_imports_work():
    import yaml  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
