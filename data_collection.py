import cv2
import mediapipe as mp
import numpy as np
import os
import time

# MediaPipe Holistic 모델 로드
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 설정값 ---
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['어깨', '목', '허리', '아프다', '몸살', '삐다']) # 학습할 액션 리스트
no_sequences = 30
sequence_length = 30

# ★★★★★ 추가: 손 감지 최소 비율 설정 ★★★★★
# 시퀀스(30프레임) 내에서 최소 이 비율 이상 손이 감지되어야 저장합니다. (0.0 ~ 1.0)
# 예: 0.5 = 50% (최소 15프레임에서 손 감지 필요)
MIN_HAND_FRAME_RATIO = 0.5
min_hand_frames = int(sequence_length * MIN_HAND_FRAME_RATIO) # 저장에 필요한 최소 손 감지 프레임 수
# ★★★★★ 추가 끝 ★★★★★

# Holistic 랜드마크 추출 함수 정의
def extract_holistic_landmarks(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) # 이전 답변에서 오타 수정됨
    return np.concatenate([pose, face, lh, rh])

# 데이터 저장 폴더 생성
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
    print(f"폴더 생성/확인 완료: {action_path}")

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit("오류: 웹캠 열기 실패")

print("\n--- 데이터 수집 준비 완료 ---")
# ... (안내 메시지 동일) ...

# Holistic 모델 초기화
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action_idx, action in enumerate(actions):
        print(f'\n--- 액션 "{action}" 데이터 수집 시작 ({action_idx+1}/{len(actions)}) ---')
        sequence_saved_count = 0 # 실제로 저장된 시퀀스 카운터
        sequence_attempt = 0 # 시도한 시퀀스 번호

        # 목표한 개수(no_sequences)만큼 저장될 때까지 시도
        while sequence_saved_count < no_sequences:
            sequence_attempt += 1
            print(f'  시퀀스 {sequence_saved_count+1}/{no_sequences} 녹화 시도 #{sequence_attempt} 준비...')

            # 'r' 키 입력 대기 (기존과 동일)
            while True:
                success, frame = cap.read()
                if not success: continue
                mirrored_frame = cv2.flip(frame, 1)
                # ★★★ 수정: 안내 메시지에 저장된 개수 표시 ★★★
                wait_text = f'Action: {action} | Saved: {sequence_saved_count}/{no_sequences} | Try #{sequence_attempt} | Press R'
                cv2.putText(mirrored_frame, wait_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', mirrored_frame)
                if cv2.waitKey(5) & 0xFF == ord('r'):
                    print(f"  시퀀스 {sequence_saved_count+1} 녹화 시작!")
                    break
                if cv2.waitKey(5) & 0xFF == ord('q'):
                     cap.release(), cv2.destroyAllWindows(), print("\n수집 중단됨."), exit()

            # --- 시퀀스 녹화 ---
            sequence_data = []
            for frame_num in range(sequence_length):
                success, frame = cap.read()
                if not success: break
                # ... (Holistic 처리 및 랜드마크 추출은 동일) ...
                frame.flags.writeable = False
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                frame.flags.writeable = True
                keypoints = extract_holistic_landmarks(results)
                sequence_data.append(keypoints)
                # ... (화면 표시 부분 동일) ...
                mirrored_frame = cv2.flip(frame, 1)
                # mp_drawing.draw_landmarks(...) # 랜드마크 그리기 (선택)
                record_text = f'RECORDING... Try #{sequence_attempt} Frame: {frame_num+1}/{sequence_length}'
                cv2.putText(mirrored_frame, record_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', mirrored_frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    cap.release(), cv2.destroyAllWindows(), print("\n수집 중단됨."), exit()

            # --- 시퀀스 저장 전 손 감지 확인 ---
            if len(sequence_data) == sequence_length:
                hand_detected_frames = 0
                # 랜드마크 인덱스 계산 (Pose: 132, Face: 1404, LH: 63, RH: 63)
                lh_start_index = 33*4 + 468*3 # 1536
                rh_start_index = lh_start_index + 21*3 # 1599

                for frame_keypoints in sequence_data:
                    # 왼쪽 손 또는 오른쪽 손 랜드마크 값이 모두 0이 아닌지 확인
                    left_hand_detected = np.any(frame_keypoints[lh_start_index : rh_start_index] != 0)
                    right_hand_detected = np.any(frame_keypoints[rh_start_index:] != 0)
                    if left_hand_detected or right_hand_detected:
                        hand_detected_frames += 1

                # 설정한 최소 프레임 수 이상 손이 감지되었는지 확인
                if hand_detected_frames >= min_hand_frames:
                    # 저장할 파일 번호는 실제로 저장된 개수 기준 (0부터 시작)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence_saved_count))
                    try:
                        np.save(npy_path, np.array(sequence_data))
                        print(f'  -> 시퀀스 {sequence_saved_count+1} 저장 완료 ({hand_detected_frames}/{sequence_length} 프레임 손 감지): {npy_path}.npy')
                        sequence_saved_count += 1 # 저장 성공 시 카운터 증가
                    except Exception as e:
                        print(f"  -> 오류: 시퀀스 {sequence_saved_count+1} 저장 실패 - {e}")
                else:
                    # 손 감지가 부족하여 저장하지 않음
                    print(f'  -> 경고: 시퀀스 저장 안 함 (시도 #{sequence_attempt}) - 손 감지 부족 ({hand_detected_frames}/{sequence_length} 프레임 < {min_hand_frames} 프레임)')
            else:
                print(f"  경고: 시퀀스 녹화 중 문제 발생({len(sequence_data)} 프레임). 저장 안 함.")
            # 다음 시도 전 잠시 대기
            # cv2.waitKey(500)

    print("\n=== 모든 액션 데이터 수집 완료! ===")
    cap.release()
    cv2.destroyAllWindows()