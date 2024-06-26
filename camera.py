import cv2


def display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = model.detectMultiScale(gray)
    for x, y, w, h in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(
            frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4
        )

    cv2.imshow("Capture - Face detection", frame)


haar = "haarcascade_frontalface_default.xml"
model = cv2.CascadeClassifier(haar)

if not model.load(haar):
    print("--(!) Error loading face cascade")
    exit(0)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("--(!) Error opening video capture")
    exit(0)

while True:
    _, frame = camera.read()

    if frame is None:
        print("--(!) No captured frame -- Break!")
        break

    display(frame)

    if cv2.waitKey(10) == 27:
        break
