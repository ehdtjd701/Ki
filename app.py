# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # .keras 모델 로드용
import os
import time
import threading # 백그라운드 작업용
import json # 라벨 저장/로드용
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

# --- 설정값 ---
# 모델 학습 시 사용된 값과 동일해야 함 (모델 로드 후 덮어쓸 수도 있음)
SEQUENCE_LENGTH = 30
NUM_FEATURES = 225 # 상대 좌표 특징 수 (포즈 33*3 + 손 21*3 * 2 = 99 + 126 = 225)

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 위치 기준
DATA_PATH = os.path.join(BASE_DIR, "gesture_data_relative") # 상대 좌표 데이터 폴더
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "gesture_model.keras") # 학습된 모델 경로
LABELS_PATH = os.path.join(DATA_PATH, "labels.json") # 라벨 매핑 파일 경로

# --- Flask 및 SocketIO 초기화 ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key' # 실제 서비스 시 변경 및 안전하게 관리
socketio = SocketIO(app)

# --- Mediapipe 초기화 ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- 전역 변수 ---
camera = cv2.VideoCapture(0)
latest_frame = None
lock = threading.Lock()
recognition_active = True # 인식 스레드 제어 플래그
last_prediction = "모델 로딩 중..." # 초기 메시지
model = None
actions = []
action_labels = {} # {action_name: index}

# --- 헬퍼 함수: 모델 및 라벨 로드 ---
def load_trained_model_and_labels():
    global SEQUENCE_LENGTH, NUM_FEATURES # 전역 변수 수정 가능하도록
    try:
        if not os.path.exists(MODEL_SAVE_PATH):
             print(f"오류: 모델 파일을 찾을 수 없습니다 - {MODEL_SAVE_PATH}")
             return None, [], {}
        if not os.path.exists(LABELS_PATH):
            print(f"오류: 라벨 파일을 찾을 수 없습니다 - {LABELS_PATH}")
            return None, [], {}

        loaded_model = load_model(MODEL_SAVE_PATH) # .keras 모델 로드
        with open(LABELS_PATH, 'r', encoding='utf-8') as f: # UTF-8 인코딩 명시
            loaded_action_labels = json.load(f)
            loaded_actions = list(loaded_action_labels.keys())

        # 모델의 입력 형태에서 SEQUENCE_LENGTH와 NUM_FEATURES를 가져올 수 있음 (선택적)
        try:
            input_shape = loaded_model.input_shape
            if len(input_shape) == 3: # (None, seq_length, num_features) 형태 확인
                SEQUENCE_LENGTH = input_shape[1]
                NUM_FEATURES = input_shape[2]
                print(f"모델 입력 형태에서 감지된 값: SEQUENCE_LENGTH={SEQUENCE_LENGTH}, NUM_FEATURES={NUM_FEATURES}")
            else:
                 print("경고: 모델 입력 형태가 예상과 다릅니다. 설정된 상수 값을 사용합니다.")
        except Exception as e:
             print(f"경고: 모델 입력 형태 확인 중 오류 ({e}). 설정된 상수 값을 사용합니다.")


        print(f"모델 로드 완료: {MODEL_SAVE_PATH}")
        print(f"로드된 라벨: {loaded_actions}")
        return loaded_model, loaded_actions, loaded_action_labels
    except Exception as e:
        print(f"모델 또는 라벨 로드 중 심각한 오류 발생: {e}")
        return None, [], {}

# --- 헬퍼 함수: 상대 좌표 계산 ---
def calculate_relative_landmarks(results):
    if not results.pose_landmarks: return np.zeros(NUM_FEATURES) # 특징 수만큼 0 반환

    landmarks = results.pose_landmarks.landmark
    lm_11 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    lm_12 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # 어깨 가시성 낮으면 0 반환
    if lm_11.visibility < 0.5 or lm_12.visibility < 0.5:
        # print("Warning: Shoulders not clearly visible.") # 너무 자주 출력될 수 있으므로 주석 처리
        return np.zeros(NUM_FEATURES)

    ref_x = (lm_11.x + lm_12.x) / 2
    ref_y = (lm_11.y + lm_12.y) / 2
    ref_z = (lm_11.z + lm_12.z) / 2

    relative_coords = []
    # 포즈 (33 * 3 = 99)
    for lm in landmarks: relative_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
    # 왼손 (21 * 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark: relative_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
    else: relative_coords.extend([0.0] * (21 * 3))
    # 오른손 (21 * 3 = 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark: relative_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
    else: relative_coords.extend([0.0] * (21 * 3))

    # NUM_FEATURES 개수 확인 (오류 방지)
    if len(relative_coords) != NUM_FEATURES:
         print(f"오류: 계산된 특징 수({len(relative_coords)})가 NUM_FEATURES({NUM_FEATURES})와 다릅니다.")
         # 길이가 다를 경우 0 배열 반환 또는 다른 처리
         return np.zeros(NUM_FEATURES)

    return np.array(relative_coords)

# --- 헬퍼 함수: 예측용 시퀀스 전처리 ---
def preprocess_for_prediction(sequence_data, seq_length=SEQUENCE_LENGTH, num_features=NUM_FEATURES):
    if not sequence_data: # 빈 시퀀스 처리
        return np.zeros((1, seq_length, num_features))

    current_len = len(sequence_data)
    if current_len >= seq_length:
        processed = np.array(sequence_data[-seq_length:])
    else:
        # 앞에 0으로 패딩
        padding = np.zeros((seq_length - current_len, num_features))
        # sequence_data가 이미 numpy 배열의 리스트일 수 있으므로 np.array로 감싸줌
        processed = np.concatenate([padding, np.array(sequence_data)])

    # 최종 형태 확인 및 모델 입력 형태로 변환
    if processed.shape != (seq_length, num_features):
         print(f"오류: 전처리 후 배열 형태({processed.shape})가 예상과 다릅니다.")
         # 예상과 다르면 0 배열 반환 (오류 방지)
         return np.zeros((1, seq_length, num_features))

    return np.expand_dims(processed, axis=0) # (1, seq_length, num_features)

# --- 모델 로드 실행 ---
model, actions, action_labels = load_trained_model_and_labels()
if model:
    last_prediction = "인식 준비 완료"
else:
    last_prediction = "오류: 모델 로드 실패"

# --- 백그라운드 동작 인식 스레드 함수 ---
def run_gesture_recognition():
    global latest_frame, lock, recognition_active, last_prediction, model, actions

    sequence_data = []
    is_recognizing_gesture = False
    last_detection_time = 0
    detection_cooldown = 1.0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while recognition_active:
            if not camera.isOpened():
                print("카메라 연결 끊김. 스레드 종료.")
                break

            success, frame = camera.read()
            if not success:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # 손 올림/내림 감지
            hands_raised = False
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                lm_l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                lm_r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lm_l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                lm_r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                if (lm_l_shoulder.visibility > 0.3 and lm_l_wrist.visibility > 0.3) or \
                   (lm_r_shoulder.visibility > 0.3 and lm_r_wrist.visibility > 0.3):
                   is_left_raised = lm_l_wrist.y < lm_l_shoulder.y - 0.05 if lm_l_shoulder.visibility > 0.3 and lm_l_wrist.visibility > 0.3 else False
                   is_right_raised = lm_r_wrist.y < lm_r_shoulder.y - 0.05 if lm_r_shoulder.visibility > 0.3 and lm_r_wrist.visibility > 0.3 else False
                   hands_raised = is_left_raised or is_right_raised

            current_time = time.time()

            # 동작 시작/종료 및 예측
            if hands_raised and not is_recognizing_gesture and (current_time - last_detection_time > detection_cooldown):
                is_recognizing_gesture = True
                sequence_data = []
                print("Gesture detection started...")
                last_prediction = "..."
                socketio.emit('recognized_text', {'text': last_prediction})

            elif not hands_raised and is_recognizing_gesture:
                is_recognizing_gesture = False
                last_detection_time = current_time
                print(f"Gesture detection ended ({len(sequence_data)} frames)")

                if model and actions and len(sequence_data) >= 10:
                    try:
                        input_data = preprocess_for_prediction(sequence_data)

                        # 입력 데이터 형태 재확인 (디버깅용)
                        # print(f"Model input shape: {input_data.shape}")

                        # 모델 예측
                        prediction = model.predict(input_data)[0]
                        predicted_index = np.argmax(prediction)
                        confidence = prediction[predicted_index]

                        # 결과 처리
                        if confidence > 0.65: # 신뢰도 임계값
                            predicted_label = actions[predicted_index]
                            last_prediction = f"{predicted_label} ({confidence*100:.1f}%)"
                        else:
                            last_prediction = "Unknown"

                        print(f"Prediction: {last_prediction}")
                        socketio.emit('recognized_text', {'text': last_prediction})

                    except Exception as e:
                        print(f"Prediction error: {e}")
                        last_prediction = "Prediction Error"
                        socketio.emit('recognized_text', {'text': last_prediction})

                elif not model or not actions:
                    last_prediction = "Model/Labels not ready"
                    socketio.emit('recognized_text', {'text': last_prediction})
                else:
                    last_prediction = "Sequence too short"
                    socketio.emit('recognized_text', {'text': last_prediction})

                sequence_data = []

            # 데이터 기록 중
            if is_recognizing_gesture:
                try:
                    relative_landmarks = calculate_relative_landmarks(results)
                    # 유효한 랜드마크만 추가 (모두 0인 경우 제외)
                    if np.any(relative_landmarks): # 0이 아닌 값이 하나라도 있으면 추가
                        sequence_data.append(relative_landmarks)
                except Exception as e:
                    print(f"Error calculating landmarks: {e}")

            # 스트리밍 프레임 업데이트
            with lock:
                 # 프레임에 랜드마크 그리기 (결과 시각화)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                          )
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                           mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                           )
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                           mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                           )
                latest_frame = frame.copy()

            socketio.sleep(0.01) # CPU 사용량 관리

    print("Recognition thread finished.")

# --- 비디오 스트리밍 함수 ---
def generate_frames():
    global latest_frame, lock
    while True:
        with lock:
            if latest_frame is None:
                # 초기 프레임 로딩 시간 벌기
                time.sleep(0.1)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", latest_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')
        socketio.sleep(0.03) # 프레임 속도 제어

# --- Flask 라우트 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- SocketIO 이벤트 핸들러 ---
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('recognized_text', {'text': last_prediction}) # 연결 시 마지막 상태 전송

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# --- 메인 실행 ---
if __name__ == '__main__':
    print("Starting background gesture recognition thread...")
    recognition_active = True
    thread = threading.Thread(target=run_gesture_recognition, daemon=True)
    thread.start()

    print("Starting Flask-SocketIO server...")
    # eventlet 또는 gevent 사용 권장 (pip install eventlet)
    # socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False) # 배포 시 debug=False
    try:
        # 개발 시에는 debug=True, use_reloader=False (스레드 중복 실행 방지)
         socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True) # Werkzeug 2.3+ 호환성
    except KeyboardInterrupt:
        print("Ctrl+C detected. Shutting down...")
    finally:
        print("Stopping recognition thread and releasing camera...")
        recognition_active = False # 스레드 종료 신호
        if thread.is_alive():
             thread.join(timeout=2) # 스레드가 끝나기를 최대 2초 기다림
        if camera.isOpened():
            camera.release()
            print("Camera released.")
        print("Server stopped.")