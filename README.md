# Akbank DL Bootcamp Projesi — GTSRB Trafik İşaretleri

Bu repo, Kaggle üzerinde geliştirdiğim derin öğrenme çalışmasının GitHub sürümüdür. 
Notebook odaklı, tekrarlanabilir ve anlaşılır bir yapı hedeflenmiştir.

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

---

## Proje Yapısı
```
akbank_dl_project/
├─ notebooks/
│  └─ DL_Egitim_1.ipynb
├─ src/
│  ├─ train.py
│  ├─ model.py
│  └─ utils.py
├─ reports/
│  └─ figures/
├─ models/
├─ logs/
├─ assets/
├─ .gitignore
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Hızlı Başlangıç
```bash
# Ortam
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Notebook
jupyter notebook notebooks/DL_Egitim_1.ipynb
```

## Reprodüksiyon
- Notebook'ta tüm hücreler sırayla çalıştırılabilir yapıdadır.
- Ağırlık dosyaları büyük olduğu için sürüme dahil edilmemiştir.

## Katkı
Issue/PR açabilirsiniz.

---

### Kaggle Notebook Linki
https://www.kaggle.com/code/haticehsnazdemir/akbank-derin-renme-bootcamp-projesi
