import cv2
import easyocr

print("Loading OCR model...")

reader = easyocr.Reader(['en'], gpu=False )

print("OCR model loaded.")


def read_from_camera():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Camera could not be opened.")
        return "I could not access the camera."

    print("Camera started.")
    print("Put text in front of camera.")
    print("Press SPACE to capture.")
    print("Press Q to quit.")

    while True:

        ret, frame = camera.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break

        cv2.imshow("Nova OCR Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        # SPACE = capture
        if key == ord(" "):

            print("Reading text...")

            results = reader.readtext(frame)

            texts = []

            for result in results:

                text = result[1]
                confidence = result[2]

                print(
                    f"Detected: {text} "
                    f"(confidence: {confidence:.2f})"
                )

                if confidence > 0.40:
                    texts.append(text)

            camera.release()
            cv2.destroyAllWindows()

            if texts:
                return " ".join(texts)

            return "I couldn't find any text."

        # Q = quit
        elif key == ord("q"):

            camera.release()
            cv2.destroyAllWindows()

            return "Camera cancelled."


# IMPORTANT
# This starts the camera when you run:
# python ocr.py

if __name__ == "__main__":

    result = read_from_camera()

    print("\n========== OCR RESULT ==========")
    print(result)
    print("================================")