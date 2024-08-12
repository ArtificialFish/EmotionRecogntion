import cv2

import torch

from PIL import Image

from model import Model
from utils import *
from dataset import ImageStandardizer


def emotion_detect(face):
    output = model(face)
    return predictions(output.data)


def standardize(face_arr):
    standardizer = ImageStandardizer()

    standardizer.fit(face_arr)
    face_arr = standardizer.transform(face_arr)

    face_arr = np.array([face_arr, face_arr, face_arr])  # 3x1x64x64

    face_arr = face_arr.transpose(1, 0, 2, 3)  # 1x3x64x64

    return face_arr


def display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Matlike -> Matlike
    gray = cv2.equalizeHist(gray)  # Matlike -> Matlike

    faces = face_model.detectMultiScale(gray)  # Matlike -> Sequence

    face_img = Image.fromarray(gray).resize((64, 64))  # Image

    X = []
    X.append(face_img)

    X = np.array(X, dtype=np.float32)
    X = standardize(X)
    X = torch.from_numpy(X)

    emotion = emotion_detect(X)

    emotion_text = emotion_labels[emotion]

    for x, y, w, h in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(
            frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4
        )

        cv2.putText(
            frame,
            emotion_text,
            (x + w + 50 // 2, y + h // 2),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (0, 0, 0),
            2,
        )

    cv2.imshow("Capture - Face detection", frame)


# GOT: numpy.ndarray, Parameter, Parameter, tuple, tuple, tuple, int


emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

haar = "haarcascade_frontalface_default.xml"
face_model = cv2.CascadeClassifier(haar)

if not face_model.load(haar):
    print("--(!) Error loading face cascade")
    exit(0)

model = Model()
model, _, _ = restore_checkpoint(model, "./checkpoints/target/", force=True)


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
