import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca data dari file CSV
data = pd.read_csv('sales_data.csv')
print(data.head())

# Mengecek data yang hilang
print(data.isnull().sum())

# Mengisi data yang hilang dengan nilai 0
data.fillna(0, inplace=True)

# Mengecek data yang hilang
print(data.isnull().sum())

# Mengubah format tanggal
data['tanggal'] = pd.to_datetime(data['tanggal'])

# Menambah kolom baru jika diperlukan, misalnya menghitung laba
data['laba'] = data['pendapatan'] - (data['jumlah'] * 8000)  # Contoh biaya produksi

# Visualisasi penjualan produk
plt.figure(figsize=(10, 6))
sns.lineplot(x='tanggal', y='pendapatan', hue='produk', data=data)
plt.title('Pendapatan Harian per Produk')
plt.xlabel('Tanggal')
plt.ylabel('Pendapatan')
plt.legend(title='Produk')
plt.show()

# Statistik deskriptif
print(data.describe())

# Menyiapkan data untuk model
X = data[['jumlah']]
y = data['pendapatan']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Scatter plot dengan garis regresi
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Aktual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Garis Regresi')
plt.title('Prediksi vs Aktual Pendapatan')
plt.xlabel('Jumlah')
plt.ylabel('Pendapatan')
plt.legend()
plt.show()

# Histogram pendapatan
plt.figure(figsize=(10, 6))
plt.hist(data['pendapatan'], bins=20, color='blue', edgecolor='black')
plt.title('Distribusi Pendapatan')
plt.xlabel('Pendapatan')
plt.ylabel('Frekuensi')
plt.show()

# Pie chart komposisi produk
produk_counts = data['produk'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(produk_counts, labels=produk_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Komposisi Produk')
plt.show()

# Bar chart pendapatan per produk
pendapatan_per_produk = data.groupby('produk')['pendapatan'].sum()
plt.figure(figsize=(10, 6))
pendapatan_per_produk.plot(kind='bar', color='skyblue')
plt.title('Total Pendapatan per Produk')
plt.xlabel('Produk')
plt.ylabel('Total Pendapatan')
plt.show()
