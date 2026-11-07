from pathlib import Path

labels = Path(r".\datasets\UA-DETRAC\train\labels")

best_file = None
best_area = 0

for f in labels.glob("*.txt"):
    for line in f.read_text().splitlines():
        a = line.split()

        if len(a) == 5 and a[0] == "2":
            area = float(a[3]) * float(a[4])

            if area > best_area:
                best_area = area
                best_file = f

print("BEST CLASS 2 SAMPLE:", best_file)
print("LARGEST BOX AREA:", best_area)
