"""The FlashInfer source pin must include the Python Mamba provider."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_flashinfer_pin_is_provider_checkout():
    pins = (ROOT / "third_party/PINS").read_text()
    setup = (ROOT / "setup.py").read_text()
    commit = re.search(r"^FLASHINFER_COMMIT=(\w{40})$", pins, re.MULTILINE)
    assert commit is not None
    assert "FLASHINFER_MAMBA_COMMIT" not in pins
    assert 'git_repository="https://github.com/yeahdongcn/flashinfer.git"' in setup
    assert "_install_flashinfer_mamba" in setup
    assert "MUSA_PROVIDER_COMMIT" in setup
    assert "shutil.copytree(source, target)" in setup
    assert "_FLASHINFER_REPO.git_tag" in setup


def test_mamba_provider_pin_is_documented():
    docs = (ROOT / "docs/mdm-developer-guide.md").read_text()
    assert "Mamba2/SSD" in docs
    assert "FLASHINFER_COMMIT" in docs


def test_docker_image_verifies_provider_installation():
    dockerfile = (ROOT / "docker/musa.Dockerfile").read_text()
    assert "FlashInfer Mamba provider installed" in dockerfile
    assert "flashinfer.mamba" in dockerfile
