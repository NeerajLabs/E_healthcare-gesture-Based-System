import cv2
import mediapipe as mp
import pyttsx3
import threading
import math
import time


# Setup speech engine
engine = pyttsx3.init()
last_spoken = ""

gesture_start_time = 0
last_detected_gesture = ""
stable_gesture = ""


def speak_non_blocking(text):
    global last_spoken
    if text != last_spoken:
        last_spoken = text
        threading.Thread(target=speak, args=(text,), daemon=True).start()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_TIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky

def distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_finger_folded(lm, tip_id):
    # Finger is folded if tip is below PIP joint (tip_id - 2)
    return lm[tip_id].y > lm[tip_id - 2].y

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                fingers_folded = sum(is_finger_folded(lm, tip_id) for tip_id in FINGER_TIPS)

                # Thumb folded check for closed fist
                thumb_tip = lm[4]
                thumb_ip = lm[3]
                thumb_folded = thumb_tip.y > thumb_ip.y

                # Distance between thumb tip and index fingertip for circle gesture
                circle_dist = distance(thumb_tip, lm[8])
                circle_threshold = 0.05  # Adjust threshold if needed
                thumb_bent = lm[4].y < lm[3].y
                index_bent = lm[8].y < lm[6].y
                middle_bent = lm[12].y > lm[10].y
                ring_bent = lm[16].y > lm[14].y
                pinky_bent = lm[20].y > lm[18].y
                index_curved_x = lm[8].x < lm[6].x
                thumb_curved_up = lm[4].y < lm[2].y
                
                is_c_shape = (
                    thumb_bent and index_bent and middle_bent and ring_bent and pinky_bent and
                    index_curved_x and thumb_curved_up
                )
                
                # Detect finger states
                index_extended = not is_finger_folded(lm, 8)
                middle_extended = not is_finger_folded(lm, 12)
                ring_extended = not is_finger_folded(lm, 16)
                pinky_extended = not is_finger_folded(lm, 20)
                
                # Check if thumb is extended
                thumb_extended = thumb_tip.y < thumb_ip.y
                
                # "Food and Water" gesture - only thumb and pinky extended, others folded
                food_water_gesture = thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended
                
                # V sign for "Yes" - only index and middle fingers extended
                v_sign_detected = index_extended and middle_extended and not ring_extended and not pinky_extended
                
                # Three fingers for "No" - index, middle, and ring fingers extended
                three_fingers_detected = index_extended and middle_extended and ring_extended and not pinky_extended
                
                # L-shape for "Bathroom" - index finger up and thumb extended horizontally
                # We need to check both extension and position to ensure L shape
                l_shape_detected = False
                if index_extended and thumb_extended and not middle_extended and not ring_extended and not pinky_extended:
                    # Additional check for L shape - thumb should be more horizontal than vertical
                    # and index finger should be more vertical than horizontal
                    
                    # Get thumb vector (from MCP to tip)
                    thumb_mcp = lm[1]  # Thumb MCP joint
                    thumb_vec_x = thumb_tip.x - thumb_mcp.x
                    thumb_vec_y = thumb_tip.y - thumb_mcp.y
                    
                    # Get index vector (from MCP to tip)
                    index_mcp = lm[5]  # Index MCP joint
                    index_vec_x = lm[8].x - index_mcp.x
                    index_vec_y = lm[8].y - index_mcp.y
                    
                    # Calculate absolute horizontal and vertical components
                    thumb_horiz = abs(thumb_vec_x)
                    thumb_vert = abs(thumb_vec_y)
                    index_horiz = abs(index_vec_x)
                    index_vert = abs(index_vec_y)
                    
                    # Check if thumb is more horizontal and index is more vertical
                    if thumb_horiz > thumb_vert and index_vert > index_horiz:
                        l_shape_detected = True
                
                if l_shape_detected:
                    # L shape is "Bathroom"
                    gesture = "Bathroom"
                    
                elif food_water_gesture:
                    # Thumb and pinky extended is "Food and Water"
                    gesture = "Food and Water"
                    
                elif v_sign_detected:
                    # V-sign is "Yes"
                    gesture = "Yes"
                    
                elif three_fingers_detected:
                    # Three fingers up is "No"
                    gesture = "No"

                # Open palm: no fingers folded, thumb extended (tip above IP joint)
                elif fingers_folded == 0 and thumb_tip.y < thumb_ip.y:
                    gesture = "Need blanket"
                # Closed fist: all fingers folded including thumb folded
                elif fingers_folded == 4 and not thumb_folded:
                    gesture = "In pain"
                # Circle gesture: thumb and index fingertips close
                elif circle_dist < circle_threshold:
                    gesture = "Okay"
                elif is_c_shape:
                    gesture = "Call Nurse"
                else:
                    gesture = ""


                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        current_time = time.time()

        if gesture != last_detected_gesture:
           gesture_start_time = current_time
           last_detected_gesture = gesture

        # Check if gesture is stable for 0.5 seconds
        if current_time - gesture_start_time > 0.5:
           if gesture != stable_gesture:
              stable_gesture = gesture
              speak_non_blocking(stable_gesture)


        cv2.imshow("E-Health Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("E-Health Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
