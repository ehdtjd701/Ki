import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# --- 설정값 ---
MODEL_PATH = 'action_model.h5'
# ★★★★★ train_model_from_npy.py 와 동일하게 설정 ★★★★★
actions = np.array(['어깨', '목', '허리', '아프다', '몸살', '삐다']) # 또는 ['아프다', '가만히']
sequence_length = 30
threshold = 0.5 # 예측 확신도 임계값

# ★★★★★ 추가: 예측 전 손 감지 확인 설정 ★★★★★
# sequence_length (예: 30) 프레임 중 최소 몇 프레임 이상 손이 감지되어야 예측을 수행할지 설정
MIN_HAND_FRAMES_FOR_PREDICTION = 10 # 예: 30프레임 중 10프레임 이상 손 감지 필요 (값 조절 가능)
# ★★★★★ 추가 끝 ★★★★★

# ★★★ 한글 폰트 설정 (사용자 환경에 맞게 수정) ★★★
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'
try:
    font_kr = ImageFont.truetype(FONT_PATH, 30)
    font_kr_small = ImageFont.truetype(FONT_PATH, 20)
    print(f"폰트 로드 성공: {FONT_PATH}")
except IOError:
    print(f"오류: 폰트 파일 '{FONT_PATH}' 로드 실패.")
    font_kr, font_kr_small = None, None
# --- 설정값 끝 ---

# --- MediaPipe 및 모델 로드 ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if not os.path.exists(MODEL_PATH): exit(f"오류: 모델 파일 '{MODEL_PATH}' 없음.")
try:
    model = load_model(MODEL_PATH)
    print(f"모델 로드 성공: {MODEL_PATH}")
except Exception as e: exit(f"오류: 모델 로드 실패 - {e}")

# --- 실시간 추론 준비 ---
sequence = []
sentence = []
predictions = []
current_action = ''
confidence = 0.0

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit("오류: 카메라 열기 실패.")

print("\n카메라 준비 완료...")
print(f"인식 대상 액션: {actions}")
print(f"최소 손 감지 프레임 (예측 전): {MIN_HAND_FRAMES_FOR_PREDICTION}/{sequence_length}")
print("'q' 키 누르면 종료.")

# Holistic 랜드마크 추출 함수
def extract_holistic_landmarks(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# 실시간 처리 루프
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # MediaPipe 처리
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        frame.flags.writeable = True

        # 랜드마크 추출
        keypoints = extract_holistic_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:] # 항상 마지막 sequence_length 만큼 유지

        # 시퀀스 길이가 충분히 모였는지 확인
        if len(sequence) == sequence_length:

            # ★★★★★ 추가: 예측 전 손 감지 확인 ★★★★★
            hand_detected_frames = 0
            lh_start_index = 33*4 + 468*3
            rh_start_index = lh_start_index + 21*3

            for frame_keypoints in sequence: # 현재 sequence 버퍼(30프레임) 확인
                left_hand_detected = np.any(frame_keypoints[lh_start_index : rh_start_index] != 0)
                right_hand_detected = np.any(frame_keypoints[rh_start_index:] != 0)
                if left_hand_detected or right_hand_detected:
                    hand_detected_frames += 1

            # 설정된 최소 프레임 수 이상 손이 감지되었는지 확인
            if hand_detected_frames >= MIN_HAND_FRAMES_FOR_PREDICTION:
                # 손이 충분히 감지되었으므로 예측 수행
                try:
                    input_data = np.expand_dims(sequence, axis=0)
                    res = model.predict(input_data, verbose=0)[0]
                    predicted_index = np.argmax(res)
                    current_confidence = res[predicted_index] # 지역 변수로 confidence 사용

                    predictions.append(predicted_index)

                    # 예측 결과 해석 및 안정화
                    N = 10
                    if len(predictions) >= N and np.unique(predictions[-N:])[0] == predicted_index:
                        if current_confidence > threshold:
                            if current_action != actions[predicted_index]:
                                current_action = actions[predicted_index]
                                confidence = current_confidence # 확정된 액션의 confidence 업데이트
                                print(f"인식된 액션: {current_action} (확률: {confidence:.2f})")
                                if len(sentence) == 0 or current_action != sentence[-1]:
                                    sentence.append(current_action)
                        # threshold 미만이지만 안정적일 때 이전 액션 유지? (선택적)
                        # elif current_action != '': pass
                    # 예측 불안정 시 이전 액션 유지 또는 초기화? (선택적)
                    elif len(predictions) >= N:
                         if current_action != '':
                             print("예측 불안정. 이전 액션 유지 또는 초기화.")
                             # current_action = '' # 필요시 초기화

                    if len(sentence) > 10: sentence = sentence[-10:]

                except Exception as e:
                    print(f"예측 중 오류 발생: {e}")
                    current_action = ''
                    confidence = 0.0
            else:
                # ★★★★★ 손이 충분히 감지되지 않으면 예측 안 함 ★★★★★
                # 이전에 인식된 액션이 있었다면 초기화
                if current_action != '':
                    print(f"손 감지 부족 ({hand_detected_frames}/{sequence_length} < {MIN_HAND_FRAMES_FOR_PREDICTION}). 액션 초기화.")
                    current_action = ''
                    confidence = 0.0 # confidence 도 리셋
                    predictions = [] # 예측 기록도 리셋하는 것이 좋을 수 있음
                # (디버깅용) print("손 감지 부족으로 예측 건너<0xEB><0x9B><0x81>")
            # ★★★★★ 추가 끝 ★★★★★

        # 화면 출력 (Pillow 사용)
        mirrored_frame = cv2.flip(frame, 1)
        # (선택 사항) 랜드마크 그리기
        # ...

        frame_to_show = mirrored_frame
        if font_kr and font_kr_small:
            try:
                img_pil = Image.fromarray(cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                # 문장 표시
                sentence_text = ' '.join(sentence)
                text_bbox = draw.textbbox((0,0), sentence_text, font=font_kr)
                text_width = text_bbox[2] - text_bbox[0]; text_height = text_bbox[3] - text_bbox[1]
                text_x = (mirrored_frame.shape[1] - text_width) // 2
                text_y = mirrored_frame.shape[0] - text_height - 10
                draw.rectangle((0, text_y - 5, mirrored_frame.shape[1], mirrored_frame.shape[0]), fill='black')
                draw.text((text_x, text_y), sentence_text, font=font_kr, fill=(255, 255, 255))
                # 상태 표시
                status_text = f'Action: {current_action} (Conf: {confidence:.2f})'
                draw.text((10, 10), status_text, font=font_kr_small, fill=(0, 255, 0))
                frame_to_show = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e: print(f"Pillow 텍스트 그리기 오류: {e}") # 오류 발생해도 중단 안되게
        elif not (font_kr and font_kr_small): # 폰트 로드 실패 시
             cv2.rectangle(mirrored_frame, (0, mirrored_frame.shape[0] - 40), (mirrored_frame.shape[1], mirrored_frame.shape[0]), (0, 0, 0), -1)
             cv2.putText(mirrored_frame, 'FONT NOT FOUND', (10, mirrored_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
             cv2.putText(mirrored_frame, f'Action: {current_action} (Conf: {confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
             frame_to_show = mirrored_frame

        cv2.imshow('Real-time Sign Language Recognition', frame_to_show)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("카메라와 창 리소스를 모두 해제했습니다.")