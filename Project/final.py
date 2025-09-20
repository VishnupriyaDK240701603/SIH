from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import random
import time
import math

app = Flask(__name__)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Liveness actions
actions = ["Blink Twice", "Blink Thrice", "Shake Head", "Nod Head"]
required_actions = random.sample(actions, 3)
action_index = 0
action_done = [False]*3

# Counters
blink_counter = 0
prev_blink_time = 0
prev_head_pos = None
shake_counter = 0
nod_counter = 0

# Verification flag
verified = False

# Eye aspect ratio helper
def eye_aspect_ratio(landmarks, eye_indices):
    top = landmarks[eye_indices[1]]
    bottom = landmarks[eye_indices[5]]
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    hor = math.dist([left.x, left.y], [right.x, right.y])
    ver = math.dist([top.x, top.y], [bottom.x, bottom.y])
    return ver / hor if hor != 0 else 0

# Action detection
def detect_actions(landmarks):
    global blink_counter, prev_blink_time, prev_head_pos, shake_counter, nod_counter

    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [263, 387, 385, 362, 380, 373]

    ear_avg = (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye)) / 2

    blink_threshold = 0.2
    current_time = time.time()
    if ear_avg < blink_threshold and current_time - prev_blink_time > 0.3:
        blink_counter += 1
        prev_blink_time = current_time

    nose_tip = landmarks[1]  # Nose tip
    head_status = ""
    if prev_head_pos:
        dx = nose_tip.x - prev_head_pos[0]
        dy = nose_tip.y - prev_head_pos[1]

        if abs(dx) > 0.03:
            shake_counter += 1
            head_status = "Shaking"
        if abs(dy) > 0.03:
            nod_counter += 1
            head_status = "Nodding"

    prev_head_pos = (nose_tip.x, nose_tip.y)

    return blink_counter, shake_counter, nod_counter, head_status

# Video streaming generator
def generate_frames():
    global action_index, action_done, blink_counter, shake_counter, nod_counter, verified

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        human_detected = False
        current_action = required_actions[action_index] if action_index < len(required_actions) else ""

        if results.multi_face_landmarks:
            human_detected = True
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                )

                blink_count, shake_count, nod_count, head_status = detect_actions(face_landmarks.landmark)

                # Check current action
                if current_action == "Blink Twice" and blink_count >= 2:
                    action_done[action_index] = True
                elif current_action == "Blink Thrice" and blink_count >= 3:
                    action_done[action_index] = True
                elif current_action == "Shake Head" and shake_count >= 2:
                    action_done[action_index] = True
                elif current_action == "Nod Head" and nod_count >= 2:
                    action_done[action_index] = True

                # Move to next action
                if action_index < len(required_actions) and action_done[action_index]:
                    action_index += 1
                    blink_counter = 0
                    shake_counter = 0
                    nod_counter = 0

        # Verification completed
        if action_index >= len(required_actions):
            verified = True

        # Overlay messages
        if human_detected:
            cv2.putText(frame, "Human Detected", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if action_index < len(required_actions):
            cv2.putText(frame, f"Do this: {required_actions[action_index]}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        else:
            cv2.putText(frame, "Human Verified ✅", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('final.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/verification_status', methods=['POST'])
def verification_status():
    global verified
    data = request.get_json()
    if verified:
        return jsonify({"message": "Verification received, can move to next page."})
    return jsonify({"message": "Not verified yet."})

if __name__ == "__main__":
    app.run(debug=True)
