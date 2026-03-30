"""
Prepare a stable production adapter directory for serving.

The production microservices path should point to a single adapter location
instead of whatever happens to be the latest training output. This script
materializes that adapter from a local checkpoint or, if needed, from the
Hugging Face artifact repo.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_SOURCE = ROOT_DIR / "results" / "sft_model" / "checkpoint-1064"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "production_adapter"
DEFAULT_REPO_ID = os.environ.get("HF_PRODUCTION_REPO_ID", "ritwijar/SRE-Nidaan-Production")
DEFAULT_SUBDIR_CANDIDATES = (
    "safe_rlhf_retry/baseline_checkpoint-1064",
    "safe_rlhf_retry/checkpoint-1064",
    "salvage/checkpoint-1064",
    "checkpoint-1064",
    "results/sft_model/checkpoint-1064",
    "sft_model/checkpoint-1064",
)
REQUIRED_FILES = ("adapter_config.json",)
OPTIONAL_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _candidate_subdirs() -> list[str]:
    raw_value = os.environ.get("HF_PRODUCTION_SUBDIR_CANDIDATES", "")
    if not raw_value.strip():
        return list(DEFAULT_SUBDIR_CANDIDATES)
    return [part.strip().strip("/") for part in raw_value.split(",") if part.strip()]


def _find_adapter_dir(root: Path) -> Path | None:
    if all((root / filename).exists() for filename in REQUIRED_FILES):
        return root
    for candidate in sorted(root.rglob("adapter_config.json")):
        if candidate.is_file():
            return candidate.parent
    return None


def _copy_adapter(source_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for filename in (*REQUIRED_FILES, *OPTIONAL_FILES):
        source_file = source_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, output_dir / filename)
            copied_files.append(filename)

    missing = [filename for filename in REQUIRED_FILES if not (output_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required adapter files after copy: {', '.join(missing)}"
        )

    artifact_label = os.environ.get("PRODUCTION_ARTIFACT_LABEL") or source_dir.name
    manifest = {
        "artifact_label": artifact_label,
        "source_dir": str(source_dir),
        "copied_files": copied_files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _download_remote_adapter(output_dir: Path) -> Path:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    repo_id = DEFAULT_REPO_ID
    base_cache_dir = ROOT_DIR / ".cache" / "production_adapter"
    base_cache_dir.mkdir(parents=True, exist_ok=True)

    for subdir in _candidate_subdirs():
        snapshot_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                token=token,
                allow_patterns=[f"{subdir}/*"],
                local_dir=str(base_cache_dir / subdir.replace("/", "__")),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        )
        adapter_dir = _find_adapter_dir(snapshot_dir)
        if adapter_dir is None:
            continue
        _copy_adapter(adapter_dir, output_dir)
        return output_dir

    raise FileNotFoundError(
        "Unable to locate a checkpoint-1064 production adapter locally or in the configured HF repo."
    )


def main() -> None:
    local_source = Path(
        os.environ.get("PRODUCTION_SOURCE_DIR", str(DEFAULT_LOCAL_SOURCE))
    ).expanduser()
    output_dir = Path(
        os.environ.get("PRODUCTION_ADAPTER_DIR", str(DEFAULT_OUTPUT_DIR))
    ).expanduser()

    if local_source.exists():
        source_dir = _find_adapter_dir(local_source) or local_source
        _copy_adapter(source_dir, output_dir)
        print(f"PRODUCTION_ADAPTER_READY={output_dir}")
        print(f"SOURCE={source_dir}")
        return

    prepared_dir = _download_remote_adapter(output_dir)
    print(f"PRODUCTION_ADAPTER_READY={prepared_dir}")
    print(f"SOURCE=hf://{DEFAULT_REPO_ID}")


if __name__ == "__main__":
    main()
