import cv2
import mediapipe as mp
import random
import numpy as np

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the first available camera
cap = cv2.VideoCapture(0)  # NOTE: If you're using a webcam, this might be 2 or another index on your system

# Set frame width and height for the captured video
frame_width = 640
frame_height = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

hands_info = {}  # Dictionary to store hand ID and wrist positions

# Function to check if the wrist position is outside the frame
def is_hand_outside_frame(wrist_x, wrist_y):
    return wrist_x < 0 or wrist_x >= frame_width or wrist_y < 0 or wrist_y >= frame_height

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    black_frame = np.zeros_like(frame)  # Create a black frame for rendering

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    current_hand_ids = set()  # Set to track current hand IDs in the frame

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get wrist landmark position
            wrist_landmark = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x, wrist_y = int(wrist_landmark.x * frame.shape[1]), int(wrist_landmark.y * frame.shape[0])

            # If wrist is within the frame, proceed
            if not is_hand_outside_frame(wrist_x, wrist_y):
                hand_id = None  # Initialize hand ID as None

                # Check if the current hand's wrist is close to any existing hand
                for existing_id, (pos_x, pos_y, visible) in hands_info.items():
                    if abs(pos_x - wrist_x) < 50 and abs(pos_y - wrist_y) < 50:
                        hand_id = existing_id
                        break

                # If no matching hand is found, assign a new unique hand ID
                if hand_id is None:
                    hand_id = random.randint(1000, 9999)
                    # Ensure the generated ID is unique by checking against existing ones
                    while hand_id in [info[0] for info in hands_info.values()]:
                        hand_id = random.randint(1000, 9999)

                # Store hand info with the hand ID, wrist position, and visibility status
                hands_info[hand_id] = (wrist_x, wrist_y, True)

                # Draw the hand landmarks on the black frame
                mp_drawing.draw_landmarks(black_frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Display hand ID near the wrist position
                cv2.putText(black_frame, f"Hand {hand_id}", (wrist_x, wrist_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                current_hand_ids.add(hand_id)  # Add hand ID to current hands set

            else:
                # Mark hand as invisible if it goes outside the frame
                for hand_id in list(hands_info.keys()):
                    if hands_info[hand_id][2]:  # Check if the hand is still visible
                        hands_info[hand_id] = (wrist_x, wrist_y, False)

    # Cleanup: Remove hands that are outside the frame and marked as invisible
    for hand_id, (wrist_x, wrist_y, visible) in list(hands_info.items()):
        if not visible and is_hand_outside_frame(wrist_x, wrist_y):
            hands_info[hand_id] = (wrist_x, wrist_y, False)

        if is_hand_outside_frame(wrist_x, wrist_y):
            hands_info[hand_id] = (wrist_x, wrist_y, False)

    # Display the result in a window
    cv2.imshow("Hand Tracking with Persistent IDs and Occlusion", black_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources after the loop ends
cap.release()
cv2.destroyAllWindows()
