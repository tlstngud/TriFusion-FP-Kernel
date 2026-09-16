import json
import subprocess
import sys

import pytest
import trifusion_l4
from trifusion_l4.cli import main


def test_metadata_import_is_independent_of_cuda_dependencies():
    script = "import sys,trifusion_l4; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_api_names_are_discoverable():
    assert trifusion_l4.__version__ == "0.1.0"
    assert {"configure_runtime", "load_student", "InferenceEngine", "fused_mamba"} <= set(
        dir(trifusion_l4)
    )
    with pytest.raises(AttributeError):
        getattr(trifusion_l4, "unknown_api")


def test_info_is_machine_readable_without_gpu(capsys):
    assert main(["info"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["version"] == "0.1.0"
    assert "torch" in data["dependencies"]


def test_invalid_checkpoint_is_rejected_before_loading_gpu(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["predict", "--weights", str(tmp_path / "missing.pt"), "--root", str(tmp_path)])
    assert error.value.code == 2
