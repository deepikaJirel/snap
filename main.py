import cv2
import mediapipe as mp
import pyttsx3
import speech_recognition as sr

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()

def speak(text):
    """Text-to-speech function"""
    print(text)
    engine.say(text)
    engine.runAndWait()

def get_user_position():
    """Ask the user to specify the desired face position using speech input."""
    speak("Please specify the face position. You can say top left, top right, bottom left, bottom right, or center.")
    with sr.Microphone() as source:
        print("Listening for user position...")
        audio = recognizer.listen(source)
        try:
            position = recognizer.recognize_google(audio).lower()
            print(f"User said: {position}")
            return position
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Please try again.")
            return get_user_position()

def face_position_in_quadrant(face_box, width, height, tolerance=0.1):
    """
    Determine in which quadrant or position the face box lies.
    We add a tolerance to allow some leeway for the center detection.
    """
    x_center = face_box.xmin + face_box.width / 2
    y_center = face_box.ymin + face_box.height / 2

    # Add tolerance for center position
    if abs(x_center - 0.5) < tolerance and abs(y_center - 0.5) < tolerance:
        return "center"
    elif x_center < 0.5 and y_center < 0.5:
        return "top left"
    elif x_center > 0.5 and y_center < 0.5:
        return "top right"
    elif x_center < 0.5 and y_center > 0.5:
        return "bottom left"
    elif x_center > 0.5 and y_center > 0.5:
        return "bottom right"
    else:
        return "center"  # If close to the center, return center by default

def guide_user(current_position, target_position):
    """Provide guiding instructions based on the user's current face position."""
    if current_position == target_position:
        speak(f"Your face is correctly positioned in the {target_position}. Capturing the image.")
        return True  # Face is correctly positioned
    else:
        if target_position == "top left":
            if "right" in current_position:
                speak("Move left.")
            if "bottom" in current_position:
                speak("Move up.")
        elif target_position == "top right":
            if "left" in current_position:
                speak("Move right.")
            if "bottom" in current_position:
                speak("Move up.")
        elif target_position == "bottom left":
            if "right" in current_position:
                speak("Move left.")
            if "top" in current_position:
                speak("Move down.")
        elif target_position == "bottom right":
            if "left" in current_position:
                speak("Move right.")
            if "top" in current_position:
                speak("Move down.")
        elif target_position == "center":
            speak("Move to the center.")
        return False  # Face is not yet in the correct position

# Open webcam and guide user to position their face
def process_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

    # Ask the user for the desired face position
    user_position = get_user_position()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detector.process(image_rgb)

        if result.detections:
            for detection in result.detections:
                # Get face bounding box
                box = detection.location_data.relative_bounding_box
                current_position = face_position_in_quadrant(box, width, height)
                if guide_user(current_position, user_position):
                    # Face is correctly positioned, capture the image
                    mp_drawing.draw_detection(frame, detection)
                    output_image_path = "captured_image.jpg"
                    cv2.imwrite(output_image_path, frame)
                    speak(f"Image captured successfully! The image has been saved as {output_image_path}.")
                    cap.release()  # Release the webcam immediately after capturing the image
                    cv2.destroyAllWindows()
                    return  # Exit the function after capturing the image

        # Display the live feed with face detection
        cv2.imshow('Webcam', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the application
process_camera()
