import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych z pliku CSV
df = pd.read_csv('cena_metra.csv', encoding='ISO-8859-2', sep=';', engine='python')



# Utworzenie kolumny 'data' jako połączenie roku i kwartału
df['data'] = df['Rok'].astype(str) + '-' + (df['Kwartal']*3-2).astype(str).str.zfill(2) + '-01'
df['data'] = pd.to_datetime(df['data'])

# Sortowanie danych względem daty
df = df.sort_values(by='data')

# Rysowanie wykresu
plt.figure(figsize=(15, 7))
plt.plot(df['data'], df['Wartosc'], marker='o')
plt.title('Cena za 1 m2 powierzchni użytkowej budynku mieszkalnego')
plt.xlabel('Data')
plt.ylabel('Cena (zł)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(df['data'], df['Wartosc'])