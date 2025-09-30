import nni
import numpy as np
import pandas as pd
from tensorflow.keras import models, layers, optimizers
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1. NNI에서 하이퍼파라미터 가져오기
# -----------------------------
params = nni.get_next_parameter()
lstm_units1 = params.get('lstm_units1', 64)
lstm_units2 = params.get('lstm_units2', 32)
learning_rate = params.get('learning_rate', 0.001)
batch_size = params.get('batch_size', 128)

# -----------------------------
# 2. 데이터 불러오기 (예시)
# -----------------------------
df_nor = pd.read_csv('dataset/press_data_normal.csv')
use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
X = df_nor[use_col].apply(lambda x: abs(x))

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

sequence = 20
X_seq = []
for i in range(len(X_scaled) - sequence):
    X_seq.append(X_scaled[i:i+sequence])
X_seq = np.array(X_seq)

# Train/Validation split
split_idx = int(len(X_seq) * 0.8)
X_train = X_seq[:split_idx]
X_valid = X_seq[split_idx:]

# -----------------------------
# 3. LSTM Autoencoder 모델 정의
# -----------------------------
def LSTM_AE(sequence, n_features):
    lstm_ae = models.Sequential()
    lstm_ae.add(layers.LSTM(lstm_units1, input_shape=(sequence, n_features), return_sequences=True))
    lstm_ae.add(layers.LSTM(lstm_units2, return_sequences=False))
    lstm_ae.add(layers.RepeatVector(sequence))
    lstm_ae.add(layers.LSTM(lstm_units2, return_sequences=True))
    lstm_ae.add(layers.LSTM(lstm_units1, return_sequences=True))
    lstm_ae.add(layers.TimeDistributed(layers.Dense(n_features)))
    return lstm_ae

model = LSTM_AE(sequence, len(use_col))
model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate))

# -----------------------------
# 4. 학습
# -----------------------------
history = model.fit(X_train, X_train,
                    epochs=10, batch_size=batch_size,
                    validation_data=(X_valid, X_valid),
                    verbose=0)

# -----------------------------
# 5. 검증 손실로 NNI에 결과 보고
# -----------------------------
val_loss = history.history['val_loss'][-1]
nni.report_final_result(val_loss)