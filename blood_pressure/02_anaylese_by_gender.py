import pandas as pd

# Wczytanie pliku CSV
df = pd.read_csv('bodyPerformance.csv')

# Grupowanie danych według płci i obliczenie korelacji, kowariancji oraz średniej wartości dla SBP i DBP
grouped = df.groupby('gender')

for name, group in grouped:
    print(f"\nDla płci {name}:")

    # Korelacja Pearsona
    pearson_corr = group[['diastolic', 'systolic']].corr(method='pearson')
    print("\nKorelacja Pearsona pomiędzy diastolic i systolic BP:")
    print(pearson_corr)

    # Kowariancja
    covariance = group[['diastolic', 'systolic']].cov()
    print("\nKowariancja pomiędzy diastolic i systolic BP:")
    print(covariance)

    # Średnie wartości
    mean_values = group[['diastolic', 'systolic']].mean()
    print("\nŚrednie wartości:")
    print(mean_values)

