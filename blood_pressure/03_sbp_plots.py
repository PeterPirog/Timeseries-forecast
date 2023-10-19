import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie pliku CSV
df = pd.read_csv('bodyPerformance.csv')

plt.figure(figsize=(10, 6))

# Dla kobiet
plt.subplot(1, 2, 1)
plt.hist2d(df[df['gender'] == 'F']['weight_kg'], df[df['gender'] == 'F']['age'], bins=(30, 30), cmin=1, cmap='viridis')
plt.colorbar(label='Ilość obserwacji')
plt.title('Kobiety')
plt.xlabel('Masa ciała (kg)')
plt.ylabel('Wiek')

# Dla mężczyzn
plt.subplot(1, 2, 2)
plt.hist2d(df[df['gender'] == 'M']['weight_kg'], df[df['gender'] == 'M']['age'], bins=(30, 30), cmin=1, cmap='viridis')
plt.colorbar(label='Ilość obserwacji')
plt.title('Mężczyźni')
plt.xlabel('Masa ciała (kg)')
plt.ylabel('Wiek')

plt.tight_layout()
plt.show()
