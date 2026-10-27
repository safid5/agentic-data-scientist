import os
import subprocess
from run_agent import main
from unittest.mock import patch


def test_smoke_run_creates_outputs(tmp_path):
    out_dir = tmp_path / "agt_outputs"
    cmd = ["python", "run_agent.py", "--data", "data/demo.csv", "--target", "auto", "--output", str(out_dir)]
    # Run the script; it should complete successfully
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"run_agent failed: {res.stderr}"
    assert os.path.exists(str(out_dir)), "Output directory was not created"

def test_run_agent_main(tmp_path):
    with patch("sys.argv", ["run_agent.py", "--data", "data/demo.csv",
                             "--target", "auto", "--output", str(tmp_path)]):
        main()
