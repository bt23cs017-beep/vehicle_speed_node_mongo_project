from pathlib import Path

labels = Path(r".\datasets\UA-DETRAC\train\labels")
images = Path(r".\datasets\UA-DETRAC\train\images")

for f in labels.glob("*.txt"):
    if not any(line.split()[0] == "2" for line in f.read_text().splitlines() if line.strip()):
        continue

    stem = f.stem
    matches = list(images.glob(stem + ".*"))

    if matches:
        print("LABEL:", f.name)
        print("IMAGE:", matches[0].name)
        break
