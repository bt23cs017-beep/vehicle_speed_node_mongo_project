from pathlib import Path
import cv2
import numpy as np

labels = Path(r".\datasets\UA-DETRAC\train\labels")
images = Path(r".\datasets\UA-DETRAC\train\images")
out = Path(r".\datasets\UA-DETRAC\car_12_preview.jpg")

files = [
    f for f in labels.glob("*.txt")
    if any(x.strip() and x.split()[0] == "0" for x in f.read_text().splitlines())
][:12]

thumbs = []

for f in files:
    image_path = images / (f.stem + ".jpg")
    im = cv2.imread(str(image_path))

    if im is None:
        continue

    h, w = im.shape[:2]

    for line in f.read_text().splitlines():
        a = line.split()
        if a[0] != "0":
            continue

        x, y, bw, bh = map(float, a[1:5])

        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 3)

    im = cv2.resize(im, (320,240))
    thumbs.append(im)

canvas = np.ones((720,1280,3), dtype=np.uint8) * 255

for i, im in enumerate(thumbs):
    r = i // 4
    c = i % 4
    canvas[r*240:(r+1)*240, c*320:(c+1)*320] = im

cv2.imwrite(str(out), canvas)

print("PREVIEW:", out)
print("IMAGES:", len(thumbs))
