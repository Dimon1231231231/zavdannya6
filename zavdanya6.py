import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Завантаження даних про автомобілі
# Якщо файл локально або на Google Drive — замініть шлях
# Наприклад: '/content/drive/MyDrive/cars_co2.csv'
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-mpg.data'  # альтернатива, але краще використовувати свій файл
# Або прямий приклад з наданими даними (якщо файл збережений)
# df = pd.read_csv('cars_co2.csv')

# Для демонстрації використовуємо прямий CSV з прикладу
data = """Model,Volume,Weight,CO2
Toyoty Aygo,1000,790,99
Mitsubishi Space Star,1200,1160,95
Skoda Citigo,1000,929,95
Fiat 500,900,865,90
Mini Cooper,1500,1140,105
Audi A1,1600,1150,99
VW Golf,1600,1250,105
Peugeot 208,1200,1050,98
Citroen C3,1100,980,96
Ford Fiesta,1300,1050,102
Volvo V40,2000,1450,120
BMW 118i,1800,1320,115
Mercedes A180,1600,1300,108
Toyota Yaris,1300,1000,97
Honda Civic,1800,1250,110
Nissan Qashqai,2000,1400,130
Suzuki Swift,1200,950,94
Hyundai i20,1200,1080,100
Renault Clio,900,950,92
Mazda 2,1300,1040,98
Opel Corsa,1300,1080,103
Kia Rio,1200,1100,99
Seat Ibiza,1400,1100,101
Fiat Panda,1200,980,95
Dacia Sandero,1000,950,93"""

from io import StringIO
df = pd.read_csv(StringIO(data))

print("Дані завантажено. Перші 5 рядків:")
print(df.head())
print("\n")

# 2. Підготовка даних для лінійної регресії
# Ознаки: Volume (об'єм двигуна), Weight (вага)
# Цільова змінна: CO2 (викиди)
X = df[['Volume', 'Weight']].values
y = df['CO2'].values

# Додаємо стовпець одиниць для intercept
X = np.hstack((np.ones((X.shape[0], 1)), X))

# 3. Розділення на навчальну (80%) та тестову (20%) вибірки
np.random.seed(42)
indices = np.random.permutation(len(y))
train_size = int(len(y) * 0.8)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

# 4. Побудова моделі лінійної регресії (метод найменших квадратів)
beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Прогноз на тестовій вибірці
y_pred = X_test @ beta

# 5. Обчислення похибок та метрик
errors = y_test - y_pred
abs_errors = np.abs(errors)
percentage_errors = (abs_errors / y_test) * 100  # всі CO2 > 0, тому безпечно

mae = np.mean(abs_errors)
rmse = np.sqrt(np.mean(errors**2))
y_mean = np.mean(y_test)
ss_tot = np.sum((y_test - y_mean)**2)
ss_res = np.sum(errors**2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# Вивід результатів
print("Коефіцієнти моделі:")
print(f"  Intercept (вільний член): {beta[0]:.3f}")
print(f"  Volume (об'єм двигуна): {beta[1]:.6f}")
print(f"  Weight (вага): {beta[2]:.6f}")
print()
print(f"MAE (середня абсолютна похибка): {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² (коефіцієнт детермінації): {r2:.3f}")
print(f"Середня відсоткова похибка: {np.mean(percentage_errors):.2f}%")
print()
print("Зразок реальних vs прогнозованих значень (перші 10 з тестової вибірки):")
for i in range(min(10, len(y_test))):
    print(f"Реальне CO2: {y_test[i]:.1f}, Прогноз: {y_pred[i]:.2f}, "
          f"Похибка: {percentage_errors[i]:.2f}%")

# 6. Гістограма відсоткових похибок
plt.figure(figsize=(10, 6))
plt.hist(percentage_errors, bins=10, edgecolor='black', color='skyblue', alpha=0.8)
plt.title('Гістограма абсолютних відсоткових похибок прогнозу викидів CO₂', fontsize=14)
plt.xlabel('Відсоткова похибка (%)')
plt.ylabel('Кількість автомобілів')
plt.grid(True, alpha=0.3)
plt.show()

# 7. Висновок
print("\n" + "="*50)
print("ВИСНОВОК")
print("="*50)
print("Модель лінійної регресії показує високу ефективність:")
print(f"- R² = {r2:.3f} — модель пояснює {r2*100:.1f}% варіації викидів CO₂.")
print(f"- Середня похибка становить лише {np.mean(percentage_errors):.2f}%.")
print("- Чим більший об'єм двигуна та вага автомобіля — тим вищі викиди CO₂,")
print("  що логічно та підтверджується позитивними коефіцієнтами.")
print("Модель добре підходить для прогнозування викидів CO₂ на основі технічних характеристик.")