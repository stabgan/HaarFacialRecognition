# Face Recognition using Haar Cascade Classifiers
# Detects faces and eyes in a live webcam feed using OpenCV.

import os
import sys
import argparse

import cv2

# ---------------------------------------------------------------------------
# Resolve cascade XML paths relative to *this* script so the program works
# regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_FACE_CASCADE_PATH = os.path.join(_SCRIPT_DIR, "haarcascade_frontalface_default.xml")
_EYE_CASCADE_PATH = os.path.join(_SCRIPT_DIR, "haarcascade_eye.xml")


def _load_cascade(path: str) -> cv2.CascadeClassifier:
    """Load a Haar cascade XML file with proper error handling."""
    cascade = cv2.CascadeClassifier()
    # Try the explicit path first, then fall back to OpenCV's bundled data.
    if os.path.isfile(path):
        loaded = cascade.load(path)
    else:
        loaded = cascade.load(cv2.samples.findFile(path, silentMode=True))

    if not loaded or cascade.empty():
        print(f"[ERROR] Could not load cascade: {path}")
        sys.exit(1)

    return cascade


def detect(gray, frame, face_cascade, eye_cascade):
    """Detect faces and eyes, drawing rectangles on *frame*.

    Parameters
    ----------
    gray : numpy.ndarray
        Grayscale version of the current video frame.
    frame : numpy.ndarray
        Original BGR video frame (will be annotated in-place).
    face_cascade : cv2.CascadeClassifier
        Loaded face cascade classifier.
    eye_cascade : cv2.CascadeClassifier
        Loaded eye cascade classifier.

    Returns
    -------
    numpy.ndarray
        The annotated frame.
    """
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a blue rectangle around each detected face.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex, ey, ew, eh) in eyes:
            # Draw a green rectangle around each detected eye.
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame


def main():
    """Run real-time face + eye detection on a webcam feed."""
    parser = argparse.ArgumentParser(
        description="Real-time face & eye detection using Haar cascades."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--face-cascade",
        type=str,
        default=_FACE_CASCADE_PATH,
        help="Path to the face Haar cascade XML file.",
    )
    parser.add_argument(
        "--eye-cascade",
        type=str,
        default=_EYE_CASCADE_PATH,
        help="Path to the eye Haar cascade XML file.",
    )
    args = parser.parse_args()

    # Load cascades with validation.
    face_cascade = _load_cascade(args.face_cascade)
    eye_cascade = _load_cascade(args.eye_cascade)

    # Open the webcam.
    video_capture = cv2.VideoCapture(args.camera)
    if not video_capture.isOpened():
        print(f"[ERROR] Cannot open camera device {args.camera}")
        sys.exit(1)

    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret or frame is None:
                print("[WARNING] No frame captured — skipping.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canvas = detect(gray, frame, face_cascade, eye_cascade)
            cv2.imshow("Video", canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
