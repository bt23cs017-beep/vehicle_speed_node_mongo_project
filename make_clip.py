import cv2, os
p = os.path.join(os.environ["USERPROFILE"], "Downloads", "13365131_3840_2160_30fps.mp4")
c = cv2.VideoCapture(p)
fps = c.get(5)
o = cv2.VideoWriter(r".\uploads\input\real_traffic.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
n = 0
while n < int(fps * 30):
    ok, f = c.read()
    if not ok:
        break
    o.write(cv2.resize(f, (1280, 720)))
    n += 1
o.release()
print("frames written:", n)
