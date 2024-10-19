import cv2
from mtcnn import MTCNN

def real_time_face_detection():
    cap = cv2.VideoCapture(0) 
    detector = MTCNN()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        cv2.imshow('Real-Time Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_face_detection()
