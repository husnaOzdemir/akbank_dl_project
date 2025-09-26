# Akbank DL Bootcamp Projesi — GTSRB Trafik İşaretleri

Bu repo, Kaggle üzerinde geliştirdiğim Akbank Derin Öğrenme Bootcamp kapsamında  derin öğrenme çalışmasının GitHub sürümüdür. 

**Kaggle Notebook:** [https://www.kaggle.com/code/haticehsnazdemir/akbank-derin-renme-bootcamp-projesi](https://www.kaggle.com/code/haticehsnazdemir/akbank-derin-renme-bootcamp-projesi)

---

## Amaç
GTSRB veri setinde trafik işaretlerini CNN tabanlı yöntemlerle sınıflandırmak; eğitim sürecini,
değerlendirme metriklerini ve görselleştirmeleri açıkça belgelemek.

## Veri Seti
- Kaynak: Kaggle (Traffic Signs Preprocessed)
- Not: Büyük veri dosyaları reponun parçası değildir; dataset, Kaggle üzerinden erişilmelidir.

## Kullanılan Yöntemler
- CNN mimarisi (Conv/BN/ReLU/Pooling/Dropout/Dense)
- Transfer Learning (MobileNetV2, opsiyonel)
- Data Augmentation (rotation, zoom, contrast; flip sınıf semantiğine göre kapalı)
- Hiperparametre denemeleri (filters, dropout, lr, kernel size, dense units, batch size, optimizer)
- İzleme: TensorBoard

## Sonuçlar
- Eğitim/Doğrulama Accuracy-Loss grafikleri
- Test doğruluğu
- Normalize Confusion Matrix + Classification Report
- Grad-CAM görselleştirmeleri

### Kaggle Notebook Linki
https://www.kaggle.com/code/haticehsnazdemir/akbank-derin-renme-bootcamp-projesi
