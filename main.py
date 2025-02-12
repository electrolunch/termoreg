import control
import numpy as np
import matplotlib.pyplot as plt

def generate_pid_data_plant2_continuous_ranges(num_datasets=240, time_points=100, time_step=0.1,
                                                 kp_range_discrete=[1, 15],
                                                 ki_range_discrete=[1e-5, 1e-3],
                                                 kd_range_discrete=[0, 30000],
                                                 sampling_frequency_hz=1000):
    """
    Генерирует наборы данных временных рядов для разных коэффициентов ПИД-регулятора,
    для системы G(s) = 40 / (400*s + 1), адаптируя дискретные диапазоны к непрерывной симуляции.

    Аргументы:
        num_datasets (int): Количество наборов данных для генерации.
        time_points (int): Количество точек во временном ряду.
        time_step (float): Шаг времени для моделирования.
        kp_range_discrete (list): Диапазон значений для Kp (дискретный) [min, max].
        ki_range_discrete (list): Диапазон значений для Ki (дискретный) [min, max].
        kd_range_discrete (list): Диапазон значений для Kd (дискретный) [min, max].
        sampling_frequency_hz (int): Частота дискретизации цифрового регулятора в Гц.

    Возвращает:
        tuple: (временные_ряды, коэффициенты_пид)
               временные_ряды - numpy array (num_datasets, time_points)
               коэффициенты_пид - numpy array (num_datasets, 3) [Kp, Ki, Kd] (непрерывные значения)
    """

    # 1. Определим систему (объект управления) G(s) = 40 / (400*s + 1).
    plant = control.TransferFunction([40], [400, 1])

    time = np.arange(0, time_points * time_step, time_step)
    setpoint = 1  # Заданное значение (ступенчатое изменение задания)

    all_time_responses = []
    all_pid_coefficients = []

    # Пересчет диапазонов для непрерывной симуляции
    kp_range_continuous = kp_range_discrete  # Kp остается таким же
    ki_range_continuous = [ki_range_discrete[0] * sampling_frequency_hz, ki_range_discrete[1] * sampling_frequency_hz]
    kd_range_continuous = [kd_range_discrete[0] / sampling_frequency_hz, kd_range_discrete[1] / sampling_frequency_hz]

    print("Диапазоны Kp (непрерывный):", kp_range_continuous)
    print("Диапазоны Ki (непрерывный):", ki_range_continuous)
    print("Диапазоны Kd (непрерывный):", kd_range_continuous)

    for _ in range(num_datasets):
        # 2. Случайная генерация коэффициентов ПИД в ПЕРЕСЧИТАННЫХ (непрерывных) диапазонах.
        kp = np.random.uniform(kp_range_continuous[0], kp_range_continuous[1])
        ki = np.random.uniform(ki_range_continuous[0], ki_range_continuous[1])
        kd = np.random.uniform(kd_range_continuous[0], kd_range_continuous[1])
        pid_coefficients = [kp, ki, kd]
        all_pid_coefficients.append(pid_coefficients)

        # 3. Создание ПИД-регулятора с текущими коэффициентами.
        pid_controller = control.tf([kd, kp, ki], [1, 0]) 

        # 4. Создание замкнутой системы.
        closed_loop_system = control.feedback(pid_controller * plant)

        # 5. Моделирование переходного процесса.
        t, y = control.step_response(closed_loop_system * setpoint, T=time)  

        # 6. Сохранение временного ряда (отклика системы).
        all_time_responses.append(y)

    return np.array(all_time_responses), np.array(all_pid_coefficients)

if __name__ == '__main__':
    # Параметры генерации данных
    num_datasets = 240
    time_points = 2000
    time_step = 0.1
    kp_range_discrete = [1, 15]
    ki_range_discrete = [1e-5, 1e-3]
    kd_range_discrete = [0, 30000]
    sampling_frequency_hz = 1000

    # Генерация данных для новой системы с пересчитанными диапазонами
    time_series_data_plant2_cont_range, pid_params_data_plant2_cont_range = generate_pid_data_plant2_continuous_ranges(
        num_datasets=num_datasets,
        time_points=time_points,
        time_step=time_step,
        kp_range_discrete=kp_range_discrete,
        ki_range_discrete=ki_range_discrete,
        kd_range_discrete=kd_range_discrete,
        sampling_frequency_hz=sampling_frequency_hz
    )

    print("Форма массива временных рядов (для G2 с пересчитанными диапазонами):", time_series_data_plant2_cont_range.shape)
    print("Форма массива коэффициентов ПИД (для G2 с пересчитанными диапазонами):", pid_params_data_plant2_cont_range.shape)

    # Визуализация нескольких сгенерированных временных рядов для новой системы
    plt.figure(figsize=(10, 6))
    for i in range(200): # Отобразим первые 5 рядов для примера
        plt.plot(np.arange(time_points) * time_step, time_series_data_plant2_cont_range[i],
                 label=f'Kp={pid_params_data_plant2_cont_range[i][0]:.2f}, Ki={pid_params_data_plant2_cont_range[i][1]:.2f}, Kd={pid_params_data_plant2_cont_range[i][2]:.2f}')

    plt.xlabel("Время")
    plt.ylabel("Регулируемая величина (выход системы)")
    plt.title("Примеры временных рядов для G(s) = 40 / (400*s + 1) с ПИД-коэффициентами (непрерывные диапазоны)")
    # plt.legend()
    plt.grid(True)
    plt.show()

    # Теперь time_series_data_plant2_cont_range и pid_params_data_plant2_cont_range можно использовать
    # для обучения нейросети, используя ПИД диапазоны, адаптированные для непрерывной симуляции.

    data_file = 'data.jsonl'
    with open(data_file, 'w') as f:
        for pid, time_series in zip(pid_params_data_plant2_cont_range, time_series_data_plant2_cont_range):
            data = {
                'kp': pid[0],
                'ki': pid[1],
                'kd': pid[2],
                'time_series': time_series.tolist()
            }
            f.write(json.dumps(data) + '\n')
    print(f"Data has been written to {data_file}")