"""Create a source ZIP that includes a genuine, clean Git checkout and its history."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

DEFAULT_REPO = "https://github.com/KittituchW/MLE_assignment.git"


def package(repository: str, branch: str, output: Path) -> str:
    """Clone only the requested branch and archive every file, including .git."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mle-submission-") as temporary:
        checkout = Path(temporary) / "MLE_assignment"
        subprocess.run(
            ["git", "clone", "--single-branch", "--branch", branch, "--", repository, str(checkout)],
            check=True,
        )
        sha = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(checkout.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(checkout.parent).as_posix())
        with ZipFile(output) as archive:
            if archive.testzip() is not None:
                raise ValueError("Submission ZIP integrity check failed")
            if "MLE_assignment/.git/HEAD" not in archive.namelist():
                raise ValueError("ZIP does not contain the Git repository metadata")
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPO)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--output", type=Path, default=Path("MLE_assignment_submission.zip"))
    args = parser.parse_args()
    sha = package(args.repository, args.branch, args.output)
    print(f"Created {args.output.resolve()} with authentic Git history at commit {sha}")


if __name__ == "__main__":
    main()
