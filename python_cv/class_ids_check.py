import cv2, glob, os, random, numpy as np
root = r".\datasets\UA-DETRAC"
labs = [p for p in glob.glob(root + r"\**\*.txt", recursive=True) if "\\labels\\" in p]
random.seed(1); random.shuffle(labs)
crops = {0: [], 1: [], 2: [], 3: []}
for lp in labs:
    base = lp.replace("\\labels\\", "\\images\\")[:-4]
    ip = next((base + e for e in (".jpg", ".png", ".jpeg") if os.path.exists(base + e)), None)
    if not ip: continue
    img = cv2.imread(ip)
    if img is None: continue
    h, w = img.shape[:2]
    for line in open(lp):
        p = line.split()
        if len(p) < 5: continue
        c = int(p[0])
        x, y, bw, bh = map(float, p[1:5])
        if c in crops and len(crops[c]) < 8 and bw * w > 40:
            x1, x2 = max(int((x - bw/2) * w), 0), int((x + bw/2) * w)
            y1, y2 = max(int((y - bh/2) * h), 0), int((y + bh/2) * h)
            cr = img[y1:y2, x1:x2]
            if cr.size: crops[c].append(cv2.resize(cr, (120, 120)))
    if all(len(v) >= 8 for v in crops.values()): break
rows = []
for c, v in crops.items():
    v = (v + [np.zeros((120, 120, 3), np.uint8)] * 8)[:8]
    row = np.hstack(v)
    cv2.putText(row, "ID " + str(c), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    rows.append(row)
out = r".\datasets\UA-DETRAC\class_ids_check.jpg"
cv2.imwrite(out, np.vstack(rows))
print("CREATED:", out)
