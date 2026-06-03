"""
Convert absolute paths in imagenet list files to paths relative to data_path.

Before: /home/caozhangjie/run-czj/dataset/imagenet/image/n02256656_13626.JPEG 0 1 ...
After:  image/n02256656_13626.JPEG 0 1 ...

Run from repo root: python data/imagenet/fix_paths.py
"""
import os

PREFIX = "/home/caozhangjie/run-czj/dataset/imagenet/"
FILES = ["train.txt", "test.txt", "database.txt"]
DIR = os.path.dirname(os.path.abspath(__file__))


def fix_file(name):
    path = os.path.join(DIR, name)
    with open(path) as f:
        lines = f.readlines()

    fixed = []
    errors = []
    for i, line in enumerate(lines, 1):
        parts = line.rstrip("\n").split(" ", 1)
        img_path = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if img_path.startswith(PREFIX):
            img_path = img_path[len(PREFIX):]
        else:
            errors.append((i, img_path))
        fixed.append(img_path + (" " + rest if rest else "") + "\n")

    if errors:
        print(f"  WARNING: {len(errors)} lines in {name} did not match expected prefix:")
        for lineno, p in errors[:5]:
            print(f"    line {lineno}: {p}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")
        return False

    with open(path, "w") as f:
        f.writelines(fixed)
    print(f"  {name}: {len(fixed)} lines fixed ✓")
    return True


if __name__ == "__main__":
    print(f"Fixing imagenet list files in {DIR}/")
    ok = all(fix_file(name) for name in FILES)
    if ok:
        print("Done. Verify with: head -1 data/imagenet/train.txt")
    else:
        print("Fix incomplete — check warnings above.")
