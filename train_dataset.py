import control
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

data_file=r"D:\PProjects\termoreg\data.jsonl"


with open(data_file, 'r') as f:
    data = []
    for line in f:
        entry = json.loads(line)
        data.append({
            'kp': entry['kp'],
            'ki': entry['ki'],
            'kd': entry['kd'],
            'time_series': entry['time_series']
        })

pid_data_list = data


# --- Шаг 2: Подготовка данных для обучения ---
# Разделение на обучающую и тестовую выборки
train_data, test_data = train_test_split(pid_data_list, test_size=0.2, random_state=42) # 80% train, 20% test

def prepare_data_for_model(data_list):
    time_series_list = [np.array(item['time_series']) for item in data_list]
    pid_params_list = [[item['kp'], item['ki'], item['kd']] for item in data_list]
    time_series_array = np.array(time_series_list)
    pid_params_array = np.array(pid_params_list)
    return time_series_array, pid_params_array

X_train_raw, y_train = prepare_data_for_model(train_data)
X_test_raw, y_test = prepare_data_for_model(test_data)

# Нормализация временных рядов (Z-score стандартизация на обучающей выборке)
mean_val = np.mean(X_train_raw)
std_val = np.std(X_train_raw)

X_train = (X_train_raw - mean_val) / std_val
X_test = (X_test_raw - mean_val) / std_val

# Reshape для LSTM и CNN (добавляем размерность признаков = 1)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Форма X_train для LSTM:", X_train_lstm.shape) # (192, 100, 1)
print("Форма X_test для LSTM:", X_test_lstm.shape)   # (48, 100, 1)
print("Форма y_train:", y_train.shape)             # (192, 3)
print("Форма y_test:", y_test.shape)               # (48, 3)

# --- Шаг 3: Обучение LSTM модели ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(3, activation='linear')) # 3 выхода: Kp, Ki, Kd

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

lstm_history = lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0) # validation_split для мониторинга

# --- Шаг 4: Обучение CNN модели ---
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

cnn_model = Sequential()
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(3, activation='linear')) # 3 выхода: Kp, Ki, Kd

cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.summary()

cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0) # validation_split для мониторинга

# --- Шаг 5: Оценка моделей ---
lstm_loss = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
cnn_loss = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)

print(f"LSTM - Loss на тестовой выборке: {lstm_loss:.4f}")
print(f"CNN  - Loss на тестовой выборке: {cnn_loss:.4f}")

# --- Шаг 6: Визуализация результатов обучения (опционально) ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='LSTM Train Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='CNN Train Loss')
plt.plot(cnn_history.history['val_loss'], label='CNN Validation Loss')
plt.title('CNN Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.tight_layout()
plt.show()