import cv2
import dlib
import face_recognition

detector = dlib.get_frontal_face_detector()

person = face_recognition.load_image_file("person.jpg")
person_enc = face_recognition.face_encodings(person)[0]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    face_loc = []
    facec = detector(frame)
    for face1 in facec:
        x = face1.left()
        y = face1.top()
        w = face1.right()
        h = face1.bottom()
        face_loc.append((y, w, h, x))

    # face_loc=face_recognition.face_locations(frame)
    face_encoding = face_recognition.face_encodings(frame, face_loc)
    i = 0
    for face in face_encoding:
        y, w, h, x = face_loc[i]
        sonuc = face_recognition.compare_faces([person_enc], face)

        if sonuc[0] == True:
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
            cv2.putText(frame, "ahmet", (x, h + 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
            cv2.putText(frame, "unknown", (x, h + 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("Window", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# opencv ile fotoğraf çekme
"""def take_photo():
    # Webcam'i başlatın
    cap = cv2.VideoCapture(0)

    # Görüntüyü yakalayın
    ret, frame = cap.read()

    # Webcam'i kapatın
    cap.release()

    # Fotoğrafı gösterin (isteğe bağlı)
    cv2.imshow('Captured Photo', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Fotoğrafı diske kaydedin
    cv2.imwrite('ahmet.jpg', frame)

if __name__ == "__main__":
    take_photo()"""















