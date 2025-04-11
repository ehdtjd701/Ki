# 필요한 라이브러리들을 불러옵니다.
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import datetime

# --- 설정값 (사용자 환경에 맞게 조절 가능) ---

# 1. 학습 데이터(.npy 파일)가 저장된 폴더 경로
# data_collection.py 에서 데이터를 저장한 경로와 동일해야 합니다.
DATA_PATH = 'MP_Data'

# 2. 학습할 액션(수어 단어) 리스트
# ★★★★★ (가장 중요!) data_collection.py 로 데이터를 생성한 ★★★★★
# ★★★★★ MP_Data 폴더 아래의 실제 폴더 이름과 정확히 일치하게 수정하세요! ★★★★★
actions = np.array(['어깨', '목', '허리', '아프다', '몸살', '삐다']) # 예: 'none' 대신 '가만히' 폴더를 만들었다면 ['아프다', '가만히'] 로 수정

# 3. 모델 학습 관련 하이퍼파라미터
epochs = 100          # 총 학습 반복 횟수 (에포크)
batch_size = 32       # 한 번에 모델에 입력할 데이터 샘플 개수
test_split_ratio = 0.2 # 전체 데이터 중 테스트 데이터로 사용할 비율
random_seed = 42      # 데이터 분할 시 사용할 랜덤 시드

# 4. 저장될 모델 파일 이름
model_save_name = 'action_model.h5'
# --- 설정값 끝 ---


# --- 1. 데이터 로딩 (개별 .npy 파일 로딩) ---
print(f"[1단계] '{DATA_PATH}' 폴더에서 개별 .npy 데이터 로딩 시작...")
sequences, labels = [], []

# actions 리스트에 정의된 각 액션에 대해 반복
for label, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action) # 해당 액션의 데이터 폴더 경로
    if not os.path.isdir(action_path):
        print(f"경고: '{action_path}' 폴더를 찾을 수 없습니다. actions 리스트와 실제 폴더 이름을 확인하세요.")
        continue # 다음 액션으로 넘어감

    print(f"'{action}' 액션 데이터 로딩 중 (폴더: '{action_path}')...")
    # 폴더 안의 .npy 파일 목록 가져오기
    sequence_files = [f for f in os.listdir(action_path) if f.endswith(".npy")]
    if not sequence_files:
        print(f"경고: '{action_path}' 폴더에 .npy 파일이 없습니다. 데이터 수집이 제대로 되었는지 확인하세요.")
        continue

    # 각 .npy 파일 로드
    for sequence_file in sequence_files:
        file_path = os.path.join(action_path, sequence_file)
        try:
            res = np.load(file_path) # 파일 로드
            # 파일 내용 유효성 검사 (비어있거나 크기가 0인지)
            if res is None or res.size == 0:
                print(f"경고: '{file_path}' 파일이 비어 있거나 손상되었습니다. 건너<0xEB><0x9A><0x8D>니다.")
                continue
            sequences.append(res) # 유효하면 sequences 리스트에 추가
            labels.append(label) # 해당 액션의 숫자 라벨 추가 (0, 1, ...)
        except Exception as e:
            print(f"오류: '{file_path}' 로딩 실패 - {e}.")

# 로드된 데이터가 있는지 최종 확인
if not sequences:
    print(f"오류: '{DATA_PATH}'에서 유효한 학습 데이터를 로드하지 못했습니다.")
    print("1. 데이터 수집 스크립트(data_collection.py)가 정상적으로 완료되었는지 확인하세요.")
    print(f"2. '{DATA_PATH}' 경로 및 하위 액션 폴더 이름이 actions 리스트와 일치하는지 확인하세요.")
    exit()
print(f"총 {len(sequences)}개의 유효한 시퀀스 데이터 로드 완료.")


# --- 2. 데이터 전처리 ---
print("\n[2단계] 데이터 전처리 시작...")
X = np.array(sequences) # 리스트를 NumPy 배열로 변환 (모델 입력용)
y = np.array(labels)   # 리스트를 NumPy 배열로 변환
y = to_categorical(y, num_classes=len(actions)).astype(int) # 숫자 라벨 -> 원-핫 인코딩

# 데이터 형태(shape) 확인 및 정보 저장
print("X 데이터 형태 (샘플 수, 시퀀스 길이, 특징 수):", X.shape)
print("y 데이터 형태 (샘플 수, 액션 개수):", y.shape)
try:
    sequence_length = X.shape[1] # 시퀀스 길이 자동 감지
    num_features = X.shape[2]    # 특징 개수(랜드마크 좌표 수) 자동 감지
    print(f"감지된 시퀀스 길이: {sequence_length}, 특징 수: {num_features}")
    print(f"참고: 특징 수 {num_features}개 (Holistic 랜드마크 기준 약 1662개)")
except IndexError:
    print("오류: 로드된 X 데이터 형태 이상 (최소 3차원 필요). 데이터 생성 과정 오류 확인 필요.")
    exit()


# --- 3. 데이터 분할 ---
print("\n[3단계] 데이터 분할 시작 (훈련용/테스트용)...")
# 샘플 수가 너무 적으면 분할 불가 오류 발생 가능성 체크
min_samples_needed = int(1 / test_split_ratio) if test_split_ratio > 0 else 1
if X.shape[0] < min_samples_needed:
     print(f"오류: 데이터 샘플 수({X.shape[0]}개)가 너무 적어 분할 불가 (최소 {min_samples_needed}개 필요).")
     print("더 많은 데이터를 수집하세요.")
     exit()
# 훈련/테스트 데이터 분할 (stratify=y 로 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split_ratio, random_state=random_seed, stratify=y
)
print("훈련 데이터:", X_train.shape, y_train.shape)
print("테스트 데이터:", X_test.shape, y_test.shape)


# --- 4. 모델 구조 설계 (LSTM) ---
print("\n[4단계] LSTM 모델 구조 설계...")
model = Sequential([
    # 첫 번째 LSTM 층: input_shape 지정 필수
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, num_features)),
    # 중간 LSTM 층: return_sequences=True 유지
    LSTM(128, return_sequences=True, activation='relu'),
    # 마지막 LSTM 층: return_sequences=False (기본값)
    LSTM(64, return_sequences=False, activation='relu'),
    # Dense 층 추가
    Dense(64, activation='relu'),
    # 과적합 방지를 위한 Dropout (선택 사항)
    Dropout(0.5),
    # 최종 출력층: units=액션 개수, 활성화 함수=softmax (확률 출력)
    Dense(len(actions), activation='softmax')
])
model.summary() # 모델 구조 출력


# --- 5. 모델 컴파일 ---
print("\n[5단계] 모델 컴파일...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("컴파일 완료.")


# --- 6. 콜백 설정 ---
print("\n[6단계] 콜백 설정...")
# TensorBoard 로그 저장 설정
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# Model Checkpoint 설정 (가장 좋은 모델 저장)
model_checkpoint = ModelCheckpoint(model_save_name, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# Reduce Learning Rate on Plateau 설정 (학습률 자동 조절)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.00001)
# Early Stopping 설정 (조기 종료)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
callbacks_list = [tensorboard_callback, model_checkpoint, reduce_lr, early_stopping]
print(f"로그 경로: '{log_dir}', 모델 저장명: '{model_save_name}'")
print("콜백 설정 완료.")


# --- 7. 모델 훈련 ---
print(f"\n[7단계] 모델 훈련 시작 (총 {epochs} 에포크)...")
# 모델 학습 실행
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test, y_test), callbacks=callbacks_list)
print("모델 훈련 완료!")


# --- 8. 모델 평가 ---
print("\n[8단계] 모델 성능 평가...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
print(f"\n학습 완료! 모델은 '{model_save_name}'에 저장됨 (또는 학습 중 가장 성능 좋았던 모델).")
print("이제 저장된 모델 파일을 실시간 추론(realtime_inference.py) 또는 동영상 추론(predict_from_video.py) 코드에서 로드하여 사용할 수 있습니다.")