import cv2
from pathlib import Path

img = Path(r".\datasets\UA-DETRAC\train\images\MVI_20011_img00092_jpg.rf.1c17d01186f910d10d67325cb36cb03b.jpg")
lab = Path(r".\datasets\UA-DETRAC\train\labels\MVI_20011_img00092_jpg.rf.1c17d01186f910d10d67325cb36cb03b.txt")

im = cv2.imread(str(img))

if im is None:
    print("ERROR: Image not found")
    exit()

h, w = im.shape[:2]

for line in lab.read_text().splitlines():
    a = line.split()

    if len(a) == 5 and a[0] == "2":
        x = float(a[1])
        y = float(a[2])
        bw = float(a[3])
        bh = float(a[4])

        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(im, "CLASS 2", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

out = Path(r".\datasets\UA-DETRAC\class2_valid_check.jpg")
cv2.imwrite(str(out), im)

print("CREATED:", out)
