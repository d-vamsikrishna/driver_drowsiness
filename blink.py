import cv2
import numpy as np
import pygame

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Load your sound file

# Face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

first_read = True
eyes_closed_start_time = None  # Variable to store the start time when eyes are closed
eyes_closed_duration_threshold = 2  # Duration threshold for closed eyes (in seconds)

# Video Capturing by using webcam
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Convert the RGB image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Applying bilateral filters to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)
    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (1, 190, 200), 2)
            # Face detector
            roi_face = gray[y:y + h, x:x + w]
            # Image
            roi_face_clr = image[y:y + h, x:x + w]
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
            if len(eyes) >= 2:  # Check if both eyes are detected
                if first_read:
                    cv2.putText(image, "", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                1, (255, 0, 0), 2)
                else:
                    cv2.putText(image, "", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                1, (255, 255, 255), 2)
                    eyes_closed_start_time = None  # Reset the timer if eyes are open
            else:
                if first_read:
                    cv2.putText(image, "", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                1, (255, 0, 255), 2)
                else:
                    cv2.putText(image, "", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                1, (0, 0, 0), 2)
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = cv2.getTickCount() / cv2.getTickFrequency()
                    else:
                        eyes_closed_duration = (cv2.getTickCount() / cv2.getTickFrequency()) - eyes_closed_start_time
                        if eyes_closed_duration > eyes_closed_duration_threshold:
                            pygame.mixer.music.play()  # Play the sound when eyes are closed for 2 seconds

                    cv2.imshow('image', image)
                    cv2.waitKey(1)

    else:
        cv2.putText(image, "", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 255, 255), 2)
    cv2.imshow('Blink', image)
    a = cv2.waitKey(1)
    if a == ord('q'):
        break
    elif a == ord('s'):
        first_read = False

# Release the webcam
cap.release()
# Close the window
cv2.destroyAllWindows()
