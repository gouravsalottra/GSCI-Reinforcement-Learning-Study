from pathlib import Path

import modal


PROJECT_DIR = Path(__file__).resolve().parent

app = modal.App("gsci-fixed-ppo-confirmatory")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "stable-baselines3==2.9.0",
        "gymnasium==1.2.3",
        "pandas==2.3.3",
        "pyarrow==21.0.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path="/root/project", copy=True)
)


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=3600,
)
def run_fixed_ppo() -> bytes:
    import sys

    sys.path.insert(0, "/root/project")
    from ppo_experiment import run_experiment

    output_path = Path("/tmp/ppo_fixed_results.zip")
    run_experiment(
        data_dir=Path("/root/project/data"),
        output_zip=output_path,
    )
    return output_path.read_bytes()


@app.local_entrypoint()
def main() -> None:
    output_path = PROJECT_DIR / "ppo_fixed_results.zip"
    output_path.write_bytes(run_fixed_ppo.remote())
    print(f"Saved: {output_path}")
