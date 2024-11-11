import cv2
import torch
import speech_recognition as sr
import pyttsx3

# Initialize YOLOv8 (Assuming model is downloaded locally or use Hugging Face)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Alternatively load YOLOv8 model

# Initialize pyttsx3 Text-to-Speech Engine
engine = pyttsx3.init()

# Set properties for the TTS engine (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Function to speak text using pyttsx3
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Capture image using camera
def capture_image():
    cap = cv2.VideoCapture(0)  # Change the index if using an external camera
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('captured_image.jpg', frame)
        print("Image captured.")
    cap.release()

# Detect objects in the image and return results
def detect_objects(image_path):
    results = model(image_path)
    return results

# Draw bounding boxes and labels on the detected objects in the image
def draw_labels(image_path, results):
    image = cv2.imread(image_path)
    for *bbox, conf, obj_class in results.xyxy[0]:
        # Draw rectangle around object
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label the object
        label = f"{results.names[int(obj_class)]} ({conf:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the labeled image
    labeled_image_path = 'labeled_image.jpg'
    cv2.imwrite(labeled_image_path, image)

    # Display the image with labels
    cv2.imshow("Labeled Image", image)
    cv2.waitKey(1000)  # Display for 1 second (adjust if necessary)
    cv2.destroyAllWindows()

# Perform speech recognition to get user commands
def get_speech_command(prompt="Say something:"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        print(f"Recognized command: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I did not understand.")
        return None

# Determine object position
def determine_position(bbox, img_width, img_height):
    x_center, y_center = bbox[0], bbox[1]
    if x_center < img_width / 3:
        horizontal = "left"
    elif x_center > 2 * img_width / 3:
        horizontal = "right"
    else:
        horizontal = "center"

    if y_center < img_height / 3:
        vertical = "top"
    elif y_center > 2 * img_height / 3:
        vertical = "bottom"
    else:
        vertical = "center"

    return f"{vertical} {horizontal}"

# Provide instructions to move the camera
def guide_movement(current_position, target_position):
    if "left" in current_position and "right" in target_position:
        speak_text("Move the camera to the right.")
    elif "right" in current_position and "left" in target_position:
        speak_text("Move the camera to the left.")
    
    if "top" in current_position and "bottom" in target_position:
        speak_text("Move the camera down.")
    elif "bottom" in current_position and "top" in target_position:
        speak_text("Move the camera up.")
    
    if "center" in current_position and "center" != target_position:
        speak_text("Move the camera in the correct direction to align with the center.")

# Main function
def main():
    target_position = "center"  # For the test case, assume the user wants the object at the center
    object_of_interest = None

    while True:
        # Step 1: Wait for user to say "capture image" before proceeding
        speak_text("Please say 'capture image' when you are ready.")
        command = get_speech_command("Say 'capture image' to take a picture.")
        if command and "capture image" in command:
            capture_image()  # Capture the image on user's command
        else:
            speak_text("Please say 'capture image' to proceed.")
            continue

        # Step 2: Detect objects after capturing the image
        objects = detect_objects('captured_image.jpg')

        # Step 3: Announce detected objects
        detected_items = objects.pandas().xyxy[0]['name'].tolist()  # Get list of detected object names
        if detected_items:
            speak_text(f"I have detected the following objects: {', '.join(detected_items)}.")
        else:
            speak_text("No objects detected in the scene.")
            continue

        # Step 4: Label the detected objects and show them on the screen
        draw_labels('captured_image.jpg', objects)

        # Step 5: Get user input for object of interest (if not already specified)
        if not object_of_interest:
            object_of_interest = get_speech_command("Please say the object you are interested in.")

        if object_of_interest and object_of_interest in detected_items:
            # Step 6: Find the position of the object
            img = cv2.imread('captured_image.jpg')
            img_height, img_width = img.shape[:2]

            for *bbox, conf, obj_class in objects.xyxy[0]:
                if objects.names[int(obj_class)] == object_of_interest:
                    current_position = determine_position(bbox, img_width, img_height)
                    print(f"The {object_of_interest} is located at the {current_position}.")

                    # Step 7: Check if object is in the desired position
                    if current_position == target_position:
                        speak_text(f"The {object_of_interest} is in the center. You can take the picture.")
                        return  # Exit if the object is in the desired position
                    else:
                        # Step 8: Provide instructions to move the camera
                        speak_text(f"The {object_of_interest} is currently at the {current_position}.")
                        guide_movement(current_position, target_position)
                        break  # Continue looping to capture the next image and recheck
        else:
            speak_text("Object not found or recognized. Please try again.")

if __name__ == "__main__":
    main()
