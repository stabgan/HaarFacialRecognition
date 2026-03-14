"""
Face & Eye Recognition using Haar Cascade Classifiers (OpenCV)

Detects faces and eyes in real-time from a webcam feed using
pre-trained Haar cascade XML models shipped with this project.
"""

import os
import sys
import cv2


def load_cascades():
    """Load Haar cascade classifiers from the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
    eye_cascade_path = os.path.join(script_dir, "haarcascade_eye.xml")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty():
        sys.exit(f"[ERROR] Could not load face cascade from: {face_cascade_path}")
    if eye_cascade.empty():
        sys.exit(f"[ERROR] Could not load eye cascade from: {eye_cascade_path}")

    return face_cascade, eye_cascade


def detect(gray, frame, face_cascade, eye_cascade):
    """
    Detect faces and eyes in a frame.

    Parameters
    ----------
    gray : numpy.ndarray
        Grayscale version of the frame.
    frame : numpy.ndarray
        Original BGR frame (will be annotated in-place).
    face_cascade : cv2.CascadeClassifier
        Haar cascade for face detection.
    eye_cascade : cv2.CascadeClassifier
        Haar cascade for eye detection.

    Returns
    -------
    numpy.ndarray
        The annotated frame with rectangles drawn around faces and eyes.
    """
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame


def main():
    face_cascade, eye_cascade = load_cascades()

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        sys.exit("[ERROR] Cannot open webcam (device 0). Check your camera connection.")

    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print("[WARN] Failed to grab frame — skipping.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame, face_cascade, eye_cascade)
        cv2.imshow("Face Recognition — Haar Cascades", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
