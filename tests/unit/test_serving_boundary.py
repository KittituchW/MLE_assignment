import os
import subprocess
import sys
from pathlib import Path


def test_serving_import_excludes_training_dependencies() -> None:
    project = Path(__file__).parents[2]
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    code = (
        "import sys; import reco.serving; "
        "bad=[x for x in ('pandas','sklearn','mlflow') if x in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=project)
