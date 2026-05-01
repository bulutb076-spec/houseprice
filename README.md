# 🏡 Kaggle Ev Fiyatları Tahmini (House Prices Advanced Regression)

Bu proje, makine öğrenmesi dünyasının en popüler yarışmalarından biri olan [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) veri seti üzerinde geliştirilmiş uçtan uca bir Veri Bilimi ve Makine Öğrenmesi çözümüdür. 

Amacımız; Ames, Iowa'daki evlerin 79 farklı özelliğini (metrekare, garaj tipi, mahalle, yapım yılı vb.) analiz ederek **satış fiyatlarını en düşük hata payıyla tahmin etmektir.**

## 🚀 Proje İş Akışı (Pipeline)

Projede veriyi doğrudan modele vermek yerine, verinin kalitesini artıracak detaylı bir **Keşifçi Veri Analizi (EDA)** ve **Özellik Mühendisliği (Feature Engineering)** uygulanmıştır:

1. **Veri İndirme ve Yükleme:** `kagglehub` kütüphanesi ile veriler doğrudan Kaggle sunucularından çekilip Pandas DataFrame'lerine aktarıldı.
2. **Çöp Kolonların Temizlenmesi:** %80'inden fazlası boş olan (`PoolQC`, `MiscFeature`, `Alley` vb.) işlevsiz özellikler tespit edilip veri setinden tamamen silindi.
3. **Akıllı Eksik Veri Doldurma (Imputation):**
   * Evde gerçekten "olmadığı" için boş bırakılan kategorik verilere (Örn: Garaj Tipi) **'Yok'** değeri atandı.
   * Evde "olmadığı" için boş bırakılan sayısal verilere (Örn: Garaj Yapım Yılı) **0** değeri atandı.
   * Ölçülmesi unutulan sayısal özelliklere (Örn: Sokağa Cephe Uzunluğu) o kolonun **Ortalaması** atandı.
   * Unutulan kategorik özelliklere ise kasabadaki en popüler değer olan **Mod** atandı.
4. **Metinleri Matematiğe Çevirme (One-Hot Encoding):** Modelin metinleri anlayabilmesi için tüm kategorik değişkenler `pd.get_dummies` ile 1 ve 0'lardan oluşan gölge değişkenlere (Dummy Variables) dönüştürüldü. Eğitim ve test setleri `.align()` metodu ile kusursuzca hizalandı.
5. **Korelasyon Analizi ve Özellik Seçimi:** Ev fiyatı ile arasındaki matematiksel ilişki (korelasyon) **%5'in altında olan** gereksiz kolonlar tespit edilip silindi. Modelin yükü hafifletildi.
6. **Model Eğitimi (Random Forest):** Temizlenmiş veri seti `%80 Eğitim` ve `%20 Doğrulama (Validation)` olarak bölündü. Doğrusal Regresyon yerine çok daha karmaşık ilişkileri çözebilen, 100 karar ağacından oluşan **RandomForestRegressor** kullanıldı.

## 💻 Kullanılan Teknolojiler

* **Dil:** Python
* **Veri İşleme:** Pandas
* **Makine Öğrenmesi:** Scikit-Learn (`RandomForestRegressor`, `train_test_split`, `mean_absolute_error`, `r2_score`)
* **Veri Kaynağı:** Kaggle API (`kagglehub`)

## 📊 Model Performansı

Modelin başarısı, sistemin hiç görmediği %20'lik doğrulama veri seti (Validation Set) üzerinde test edilmiştir:
* **MAE (Ortalama Mutlak Hata):** Modelin ev fiyatlarını tahmin ederken ortalama ne kadar saptığını gösterir.
* **R2 Score (Başarı Oranı):** Modelin ev fiyatlarındaki değişimi yüzde kaç oranında doğru açıklayabildiğini gösterir.

*(Projeyi çalıştırdığınızda güncel metrikler terminalinizde belirecektir.)*

## 🛠️ Nasıl Çalıştırılır?

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Gerekli kütüphaneleri yükleyin:
   
```bash
   pip install pandas scikit-learn kagglehub
