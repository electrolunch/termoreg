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
lstm_model.add(LSTM(128, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True)) # Увеличенное количество нейронов + return_sequences=True
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(64, return_sequences=False)) # Добавлен второй LSTM слой
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(3, activation='linear')) # 3 выхода: Kp, Ki, Kd

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

lstm_history = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0) # Увеличено количество эпох

# --- Шаг 4: Обучение CNN модели (УВЕЛИЧЕННАЯ СЛОЖНОСТЬ) ---
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2]))) # Увеличено количество фильтров
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu')) # Увеличено количество фильтров
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu')) # Добавлен еще один Conv1D слой
cnn_model.add(MaxPooling1D(pool_size=2)) # Добавлен MaxPooling после нового Conv1D
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu')) # Увеличено количество нейронов
cnn_model.add(Dropout(0.3)) # Увеличен dropout
cnn_model.add(Dense(3, activation='linear')) # 3 выхода: Kp, Ki, Kd

cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.summary()

cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0) # Увеличено количество эпох

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

# --- Шаг 7: Визуализация предсказаний (НОВОЕ) ---
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_cnn = cnn_model.predict(X_test_cnn)

pid_labels = ['Kp', 'Ki', 'Kd']
plt.figure(figsize=(15, 5))

for i in range(3): # Для каждого параметра ПИД (Kp, Ki, Kd)
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test[:, i], y_pred_lstm[:, i], label='LSTM Predictions', alpha=0.7)
    plt.scatter(y_test[:, i], y_pred_cnn[:, i], label='CNN Predictions', alpha=0.7)
    plt.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], 'r--', label='Ideal Prediction') # Диагональ для идеального предсказания
    plt.xlabel('True ' + pid_labels[i])
    plt.ylabel('Predicted ' + pid_labels[i])
    plt.title(pid_labels[i] + ' - True vs Predicted')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
# --- Шаг 8: Предсказание модели на тестовых данных и сверка ---
# Предсказание на тестовых данных
# Rewriting the selection according to the provided instructions

# Извлечение входных данных и меток из test_data
X_test = np.array([entry['time_series'] for entry in test_data])
y_test = np.array([[entry['kp'], entry['ki'], entry['kd']] for entry in test_data])

# Предсказание с использованием моделей
predictions_lstm = lstm_model.predict(X_test)
predictions_cnn = cnn_model.predict(X_test)

for sample_index in range(1, 5):
    x_sample = X_test[sample_index]
    y_true = y_test[sample_index]
    
    # Сделать предсказания
    prediction_lstm = predictions_lstm[sample_index]
    prediction_cnn = predictions_cnn[sample_index]
    
    # Вывод результатов
    print(f"Индекс образца: {sample_index}")
    print(f"Истинные коэффициенты PID: Kp={y_true[0]}, Ki={y_true[1]}, Kd={y_true[2]}")
    print(f"LSTM Предсказание: Kp={prediction_lstm[0]:.4f}, Ki={prediction_lstm[1]:.4f}, Kd={prediction_lstm[2]:.4f}")
    print(f"CNN Предсказание: Kp={prediction_cnn[0]:.4f}, Ki={prediction_cnn[1]:.4f}, Kd={prediction_cnn[2]:.4f}")

