import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Veri dosyalarını yükle
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Veri setlerinin boyutunu kontrol et
print(f"Eğitim verisi boyutu: {train.shape}")
print(f"Test verisi boyutu: {test.shape}")

# Eğitim veri setinin ilk 5 satırını yazdır
print("Eğitim veri setinin ilk 5 satırı:")
print(train.head())

# Özellikler (X) ve etiketler (y) ayırma
X = train.drop(columns=['label'])  # Pixel verileri
y = train['label']  # Etiketler (0-9 arası rakamlar)

# Eğitim verisini eğitim ve doğrulama (validation) olarak bölelim
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri setlerini ölçeklendirelim (StandardScaler ile normalizasyon)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Test veri seti üzerinde ölçeklendirme işlemi yapalım
X_test_scaled = scaler.transform(test)  # Test verisi doğrudan kullanılıyor (ImageId sütunu yok)

# Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Modelin doğruluğunu eğitim seti üzerinde değerlendirelim
y_train_pred = model.predict(X_train_scaled)
print(f"Eğitim seti başarı oranı: {accuracy_score(y_train, y_train_pred) * 100:.2f}%")

# Eğitim seti üzerindeki tahmin sonuçlarını yazdır
print("Eğitim seti üzerindeki ilk 5 tahmin:")
print(y_train_pred[:5])

# Modeli doğrulama seti (validation) üzerinde test edelim
y_val_pred = model.predict(X_val_scaled)
print(f"Doğrulama seti başarı oranı: {accuracy_score(y_val, y_val_pred) * 100:.2f}%")

# Test seti üzerinde tahmin yapalım
y_pred = model.predict(X_test_scaled)

# Test seti üzerindeki ilk 5 tahmin
print("Test seti üzerindeki ilk 5 tahmin:")
print(y_pred[:5])

# Modeli kaydedelim (opsiyonel)
joblib.dump(model, 'digit_recognizer_model.pkl')

# Eğer test verisinin etiketleri varsa (örneğin `y_test`), doğruluğu hesaplayabiliriz:
# print(f"Test Seti Başarı Oranı: {accuracy_score(y_test, y_pred) * 100:.2f}%")




