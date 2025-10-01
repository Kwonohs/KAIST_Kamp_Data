import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model, models, layers, optimizers, regularizers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle
import nni
import os
import json

def load_and_preprocess_data():
    """데이터 로드 및 전처리 함수"""
    # 판다스 데이터 불러오기
    df_fault = pd.read_csv('dataset/outlier_data.csv')
    df_nor = pd.read_csv('dataset/press_data_normal.csv')
    
    # 데이터 copy
    normal = df_nor.copy()
    outlier = df_fault.copy()
    
    # 정상데이터 시각화 + use_col 활요하여 독립변수 추출
    use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
    
    # apply 활용하여 함수 적용
    normal[use_col] = normal[use_col].apply(lambda x: abs(x))
    outlier[use_col] = outlier[use_col].apply(lambda x: abs(x))
    
    # 입력데이터, 타겟데이터 분류.
    X_normal = normal[use_col]
    y_normal = normal['Equipment_state']
    X_anomaly = outlier[use_col]
    y_anomaly = outlier['Equipment_state']
    
    # 데이터 split
    X_train_normal = X_normal[:15000]
    y_train_normal = y_normal[:15000]
    X_test_normal = X_normal[15000:]
    y_test_normal = y_normal[15000:]
    X_test_anomaly = X_anomaly
    y_test_anomaly = y_anomaly
    
    # 입력변수 스케일링
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_normal_scaled = scaler.transform(X_test_normal)
    X_test_anomaly_scaled = scaler.transform(X_test_anomaly)
    
    # 종속변수 list -> numpy배열 변경
    y_train_normal = np.array(y_train_normal)
    y_test_normal = np.array(y_test_normal)
    y_test_anomaly = np.array(y_test_anomaly)
    
    return (X_train_scaled, X_test_normal_scaled, X_test_anomaly_scaled, 
            y_train_normal, y_test_normal, y_test_anomaly, scaler)

def create_sequences(X_data, y_data, sequence, offset):
    """시퀀스 데이터 생성 함수"""
    X_seq, Y_seq = [], []
    for index in range(len(X_data) - sequence - offset):
        X_seq.append(X_data[index: index + sequence])
        Y_seq.append(y_data[index + sequence + offset])
    return np.array(X_seq), np.array(Y_seq)

def LSTM_AE(sequence, n_features, lstm_units_1, lstm_units_2, dropout_rate):
    """LSTM Autoencoder 모델 생성 함수"""
    lstm_ae = models.Sequential()
    
    # Encoder
    lstm_ae.add(layers.LSTM(lstm_units_1, input_shape=(sequence, n_features), 
                           return_sequences=True, dropout=dropout_rate))
    lstm_ae.add(layers.LSTM(lstm_units_2, return_sequences=False, dropout=dropout_rate))
    lstm_ae.add(layers.RepeatVector(sequence))  # 시퀀스 길이 복원
    
    # Decoder
    lstm_ae.add(layers.LSTM(lstm_units_2, return_sequences=True, dropout=dropout_rate))
    lstm_ae.add(layers.LSTM(lstm_units_1, return_sequences=True, dropout=dropout_rate))
    lstm_ae.add(layers.TimeDistributed(layers.Dense(n_features)))
    
    return lstm_ae

def flatten(X):
    """3D 배열을 2D로 변환하는 함수"""
    flattened = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened[i] = X[i, X.shape[1] - 1, :]
    return flattened

def calculate_optimal_threshold(y_true, mse):
    """최적 임계값 계산 함수"""
    precision, recall, threshold = metrics.precision_recall_curve(y_true, mse)
    index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision, recall)) if p == r][0]
    threshold_final = threshold[index_cnt]
    return threshold_final, precision[index_cnt], recall[index_cnt]

def evaluate_model(model, X_test, Y_test, threshold):
    """모델 평가 함수"""
    test_x_predictions = model.predict(X_test)
    mse = np.mean(np.power(flatten(X_test) - flatten(test_x_predictions), 2), axis=1)
    
    pred_y = [1 if e > threshold else 0 for e in mse]
    
    # 성능 지표 계산
    accuracy = metrics.accuracy_score(Y_test, pred_y)
    precision = metrics.precision_score(Y_test, pred_y)
    recall = metrics.recall_score(Y_test, pred_y)
    f1 = metrics.f1_score(Y_test, pred_y)
    
    return accuracy, precision, recall, f1, mse

def main():
    """메인 실행 함수"""
    # NNI 하이퍼파라미터 받기
    params = nni.get_next_parameter()
    
    # 하이퍼파라미터 설정
    lstm_units_1 = params['lstm_units_1']
    lstm_units_2 = params['lstm_units_2']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    dropout_rate = params['dropout_rate']
    sequence = params['sequence_length']
    offset = params['offset']
    
    print(f"Trial parameters: {params}")
    
    # 데이터 로드 및 전처리
    (X_train_scaled, X_test_normal_scaled, X_test_anomaly_scaled,
     y_train_normal, y_test_normal, y_test_anomaly, scaler) = load_and_preprocess_data()
    
    # 시퀀스 데이터 생성
    X_train, Y_train = create_sequences(X_train_scaled, y_train_normal, sequence, offset)
    X_test_normal, Y_test_normal = create_sequences(X_test_normal_scaled, y_test_normal, sequence, offset)
    X_test_anomal, Y_test_anomal = create_sequences(X_test_anomaly_scaled, y_test_anomaly, sequence, offset)
    
    # 데이터 분할
    X_valid_normal, Y_valid_normal = X_test_normal[:880, :, :], Y_test_normal[:880]
    X_test_normal, Y_test_normal = X_test_normal[880:, :, :], Y_test_normal[880:]
    X_valid_anomal, Y_valid_anomal = X_test_anomal[:300, :, :], Y_test_anomal[:300]
    X_test_anomal, Y_test_anomal = X_test_anomal[300:, :, :], Y_test_anomal[300:]
    
    # 검증 및 테스트 데이터 결합
    X_valid = np.vstack((X_valid_normal, X_valid_anomal))
    Y_valid = np.hstack((Y_valid_normal, Y_valid_anomal))
    X_test = np.vstack((X_test_normal, X_test_anomal))
    Y_test = np.hstack((Y_test_normal, Y_test_anomal))
    
    # 정상 데이터만으로 검증 (Autoencoder 학습용)
    X_valid_0 = X_valid[Y_valid==0]
    
    # 모델 생성
    lstm_ae = LSTM_AE(sequence, 3, lstm_units_1, lstm_units_2, dropout_rate)
    
    # 콜백 설정
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=20, verbose=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=50, verbose=0, 
                      mode='min', restore_best_weights=True)
    
    # 모델 컴파일
    lstm_ae.compile(loss='mse', optimizer=optimizers.Adam(learning_rate))
    
    # 모델 학습
    history = lstm_ae.fit(X_train, X_train, 
                         epochs=600, 
                         batch_size=batch_size, 
                         callbacks=[reduce_lr, es], 
                         validation_data=(X_valid_0, X_valid_0),
                         verbose=0)
    
    # 최적 임계값 계산
    valid_x_predictions = lstm_ae.predict(X_valid, verbose=0)
    mse_valid = np.mean(np.power(flatten(X_valid) - flatten(valid_x_predictions), 2), axis=1)
    threshold_final, precision_val, recall_val = calculate_optimal_threshold(list(Y_valid), mse_valid)
    
    # 테스트 데이터로 최종 평가
    accuracy, precision, recall, f1, mse_test = evaluate_model(lstm_ae, X_test, Y_test, threshold_final)
    
    # NNI에 결과 보고
    nni.report_final_result({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold_final,
        'val_precision': precision_val,
        'val_recall': recall_val,
        'final_loss': history.history['val_loss'][-1]
    })
    
    print(f"Final Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

if __name__ == '__main__':
    main()
