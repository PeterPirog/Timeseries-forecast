# https://www.kaggle.com/datasets/kukuroo3/body-performance-data

"""
data shape : (13393, 12)

age : 20 ~64
gender : F,M
height_cm : (If you want to convert to feet, divide by 30.48)
weight_kg
body fat_%
diastolic : diastolic blood pressure (min)
systolic : systolic blood pressure (min)
gripForce
sit and bend forward_cm
sit-ups counts
broad jump_cm
class : A,B,C,D ( A: best) / stratified
"""

# Importowanie potrzebnych bibliotek
import pandas as pd

# Wczytanie pliku CSV
df = pd.read_csv('bodyPerformance.csv')

# Wyświetlenie pierwszych pięciu wierszy danych
print(df.head())

# Wyświetlenie informacji o zbiorze
print(df.info())

# Obliczenie i wyświetlenie korelacji Pearsona
pearson_corr = df[['diastolic', 'systolic']].corr(method='pearson')
print("\nKorelacja Pearsona pomiędzy diastolic i systolic BP:")
print(pearson_corr)

# Obliczenie i wyświetlenie kowariancji
covariance = df[['diastolic', 'systolic']].cov()
print("\nKowariancja pomiędzy diastolic i systolic BP:")
print(covariance)
