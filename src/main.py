from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc.udp_client import SimpleUDPClient


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
FRAME_MIDPOINT = 0.5  # normalized x -- left of this is "Person 1", right is "Person 2"

OSC_IP = "127.0.0.1"   # localhost -- Python and TouchDesigner run on the same machine
OSC_PORT = 8000         # must match the port on TouchDesigner's OSC In CHOP


def ensureModel():
    if MODEL_PATH.exists():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe hand model...")
    with urlopen(MODEL_URL) as response, MODEL_PATH.open("wb") as modelFile:
        modelFile.write(response.read())


def drawHand(frame, landmarks):
    frameHeight, frameWidth = frame.shape[:2]
    points = [
        (int(landmark.x * frameWidth), int(landmark.y * frameHeight))
        for landmark in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def personZoneFor(wristX):
    return "Person 1" if wristX < FRAME_MIDPOINT else "Person 2"


def assignHandLabels(handLandmarksList, handednessList):
    """
    Maps each detected hand to one of four slots:
    "Person 1 Left", "Person 1 Right", "Person 2 Left", "Person 2 Right".
    """
    labeledHands = {}

    for handLandmarks, handedness in zip(handLandmarksList, handednessList):
        wrist = handLandmarks[0]
        zone = personZoneFor(wrist.x)
        role = handedness[0].category_name  # "Left" or "Right"
        confidence = handedness[0].score
        label = f"{zone} {role}"

        existing = labeledHands.get(label)
        if existing is None or confidence > existing[2]:
            labeledHands[label] = (handLandmarks, wrist, confidence)

    return labeledHands


def drawZoneOverlay(frame):
    frameHeight, frameWidth = frame.shape[:2]
    midX = int(frameWidth * FRAME_MIDPOINT)
    cv2.line(frame, (midX, 0), (midX, frameHeight), (255, 255, 0), 2)
    cv2.putText(frame, "Person 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, "Person 2", (midX + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def drawHandLabel(frame, label, wrist):
    frameHeight, frameWidth = frame.shape[:2]
    x, y = int(wrist.x * frameWidth), int(wrist.y * frameHeight)
    cv2.putText(frame, label, (x - 40, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


def oscAddressFor(label):
    # "Person 1 Left" -> "/person1/left"
    person, number, role = label.split(" ")
    return f"/{person.lower()}{number}/{role.lower()}"


def sendHandOsc(oscClient, label, wrist):
    address = oscAddressFor(label)
    oscClient.send_message(f"{address}/x", wrist.x)
    oscClient.send_message(f"{address}/y", wrist.y)


ensureModel()
baseOptions = python.BaseOptions(model_asset_path=str(MODEL_PATH))
handLandmarkerOptions = vision.HandLandmarkerOptions(
    base_options=baseOptions,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
handLandmarker = vision.HandLandmarker.create_from_options(handLandmarkerOptions)
oscClient = SimpleUDPClient(OSC_IP, OSC_PORT)

cap = cv2.VideoCapture(0)
timestampMs = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Camera frame not received, skipping.")
        continue

    # Flip for a natural "mirror" view. MediaPipe's handedness labels
    # assume a mirrored/selfie-style image, so flip BEFORE processing.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestampMs += 1

    results = handLandmarker.detect_for_video(image, timestampMs)
    drawZoneOverlay(frame)

    if results.hand_landmarks:
        labeledHands = assignHandLabels(results.hand_landmarks, results.handedness)

        for label, (handLandmarks, wrist, confidence) in labeledHands.items():
            drawHand(frame, handLandmarks)
            drawHandLabel(frame, label, wrist)
            sendHandOsc(oscClient, label, wrist)
            print(
                f"{label}: (confidence {confidence:.2f}) "
                f"at x={wrist.x:.3f}, y={wrist.y:.3f}"
            )

    cv2.imshow("Rehab Rhythm - Hand Tracking (press q to quit)", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
handLandmarker.close()
cv2.destroyAllWindows()