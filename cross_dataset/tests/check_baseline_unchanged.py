"""Check both index and working tree against the user's frozen baseline."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "a8318b4da99879c625a60154d111a510e5ac5a91"
DOC = "docs/PandaSet与DDAD零样本泛化及训练评估接入方案.md"
ALLOWED = ("cross_dataset/", "configs/PandaSet/", "configs/DDAD/", "configs/ZeroShot/")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def check():
    errors, hashes = [], {}
    tree = git("ls-tree", "-rz", BASELINE).split(b"\0")
    for row in filter(None, tree):
        meta, encoded = row.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        name = encoded.decode()
        if kind != "blob":
            continue
        original = git("cat-file", "blob", oid)
        hashes[name] = hashlib.sha256(original).hexdigest()
        if name == DOC:
            continue
        expected = original
        if name == ".gitignore":
            expected = original.replace(b"nuScenes\n", b"nuScenes\n/data/PandaSet\n/data/DDAD\n", 1)
        path = ROOT / name
        try:
            current = os.readlink(path).encode() if mode == "120000" else path.read_bytes()
            if current != expected:
                errors.append(f"working tree content: {name}")
            if mode != "120000" and (path.is_symlink() or bool(path.stat().st_mode & 0o111) != (mode == "100755")):
                errors.append(f"working tree mode: {name}")
            staged = git("show", f":{name}")
            index_mode = git("ls-files", "-s", "--", name).decode().split()[0]
            if staged != expected or index_mode != mode:
                errors.append(f"index: {name}")
        except (OSError, subprocess.CalledProcessError):
            errors.append(f"missing: {name}")
    new_names = set(git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0"))
    new_names.update(git("diff", "--cached", "--diff-filter=A", "--name-only", "-z", BASELINE).decode().split("\0"))
    for name in sorted(new_names - {"", DOC}):
        if not name.startswith(ALLOWED):
            errors.append(f"outside allowed additions: {name}")
    for name, target in {
        "nuScenes": "/home/B_UserData/dongzhipeng/Datasets/dataset_omniscene",
        "PandaSet": "/home/B_UserData/dongzhipeng/Datasets/PandaSet",
        "DDAD": "/home/B_UserData/dongzhipeng/Datasets/DDAD",
    }.items():
        path = ROOT / "data" / name
        if not path.is_symlink() or os.readlink(path) != target:
            errors.append(f"dataset symlink: {path}")
    if errors:
        raise AssertionError("\n".join(errors))
    return {"baseline": BASELINE, "tracked_blobs_checked": len(hashes),
            "baseline_manifest_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "status": "unchanged", "exceptions": [".gitignore: two dataset entries", DOC]}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, ensure_ascii=False))
