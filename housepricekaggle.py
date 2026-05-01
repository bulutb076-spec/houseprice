import os
import kagglehub
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. VERİYİ İNDİR VE OKU
# ==========================================
print("Kaggle'dan ev verileri indiriliyor, lütfen bekleyin...")
klasor_yolu = kagglehub.competition_download('house-prices-advanced-regression-techniques')

train_dosyasi = os.path.join(klasor_yolu, 'train.csv')
test_dosyasi  = os.path.join(klasor_yolu, 'test.csv')

train_veriler = pd.read_csv(train_dosyasi)
test_veriler  = pd.read_csv(test_dosyasi)

# Test'in Id kolonunu submission için sakla
test_ids = test_veriler['Id']

# Eksik veri raporu
eksikler = train_veriler.isnull().sum()
sorunlu_kolonlar = eksikler[eksikler > 0].sort_values(ascending=False)
print("\nEksik veri içeren kolonlar:")
print(sorunlu_kolonlar)


# ==========================================
# 2. ÇÖP KOLONLARI SİL
# ==========================================
cop_kolonlar = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu']

train_veriler = train_veriler.drop(cop_kolonlar, axis=1)
test_veriler  = test_veriler.drop(cop_kolonlar, axis=1)

print(f"\nTemizlik yapıldı! Yeni kolon sayısı: {train_veriler.shape[1]}")


# ==========================================
# 3. EKSİK VERİLERİ DOLDUR
# ==========================================

# Metinler → 'Yok'
yok_ile = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
train_veriler[yok_ile] = train_veriler[yok_ile].fillna('Yok')
test_veriler[yok_ile]  = test_veriler[yok_ile].fillna('Yok')

# Sayılar → 0
sifir_ile = ['GarageYrBlt', 'MasVnrArea']
train_veriler[sifir_ile] = train_veriler[sifir_ile].fillna(0)
test_veriler[sifir_ile]  = test_veriler[sifir_ile].fillna(0)

# LotFrontage → Ortalama (train ortalaması kullanılıyor)
lot_ort = train_veriler['LotFrontage'].mean()
train_veriler['LotFrontage'] = train_veriler['LotFrontage'].fillna(lot_ort)
test_veriler['LotFrontage']  = test_veriler['LotFrontage'].fillna(lot_ort)

# Electrical → Mod
elektrik_modu = train_veriler['Electrical'].mode()[0]
train_veriler['Electrical'] = train_veriler['Electrical'].fillna(elektrik_modu)
test_veriler['Electrical']  = test_veriler['Electrical'].fillna(elektrik_modu)

print(f"\nEksik veri kaldı mı? (0 olmalı): {train_veriler.isnull().sum().sum()}")


# ==========================================
# 4. KATEGORİK VERİLERİ SAYISALLAŞTIR
# ==========================================
metin_kolonlari = train_veriler.select_dtypes(include=['object']).columns
print(f"\nToplam {len(metin_kolonlari)} adet metin kolonu dönüştürülüyor...")

train_veriler = pd.get_dummies(train_veriler, columns=metin_kolonlari)
test_veriler  = pd.get_dummies(test_veriler,  columns=metin_kolonlari)

# Train ve Test kolonlarını hizala
train_veriler, test_veriler = train_veriler.align(test_veriler, join='left', axis=1)
test_veriler = test_veriler.fillna(0)

print(f"Dönüşüm tamam! Yeni kolon sayısı: {train_veriler.shape[1]}")


# ==========================================
# 5. KORELASYON TEMİZLİĞİ
# ==========================================
print("\nFiyatla ilişkisi zayıf kolonlar temizleniyor...")

korelasyon_matrisi = train_veriler.corr()
fiyat_etkisi = korelasyon_matrisi['SalePrice'].abs()
silinecek_kolonlar = fiyat_etkisi[fiyat_etkisi < 0.05].index

train_veriler = train_veriler.drop(columns=silinecek_kolonlar)
test_veriler  = test_veriler.drop(columns=silinecek_kolonlar, errors='ignore')

print(f"{len(silinecek_kolonlar)} adet gereksiz kolon silindi.")
print(f"Kalan kolon sayısı: {train_veriler.shape[1]}")


# ==========================================
# 6. MODEL EĞİT VE DEĞERLENDİR
# ==========================================
X = train_veriler.drop('SalePrice', axis=1)
y = train_veriler['SalePrice']

# %80 Eğitim / %20 Doğrulama
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest — Linear Regression'dan çok daha güçlü
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

val_tahminleri = model.predict(X_val)

mae = mean_absolute_error(y_val, val_tahminleri)
r2  = r2_score(y_val, val_tahminleri)

print(f"\n--- MODEL PERFORMANS RAPORU ---")
print(f"Ortalama Hata   (MAE): {mae:,.2f} Dolar")
print(f"Başarı Oranı  (R²):  %{r2 * 100:.2f}")


# ==========================================
# 7. KAGGLE SUBMISSION DOSYASI OLUŞTUR
# ==========================================

# Tüm eğitim verisiyle modeli yeniden eğit (val kısmı da dahil olsun)
model.fit(X, y)

X_test = test_veriler.drop('SalePrice', axis=1, errors='ignore')
tahminler = model.predict(X_test)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': tahminler
})
submission.to_csv('submission.csv', index=False)

print(f"\n✅ submission.csv oluşturuldu! ({len(submission)} satır)")
print("Kaggle'a yükleyebilirsin.")