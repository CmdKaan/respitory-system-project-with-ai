import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Proje:
    def __init__(self, dosya_yolu, TSR=16000):
        self.TSR = TSR
        self.pre_emphasis_katsayisi = 0.97
        self.signal, _ = librosa.load(dosya_yolu, sr=self.TSR)
        max_val = np.max(np.abs(self.signal))
        if max_val > 0:
            self.signal = self.signal / max_val
        self.signal = np.append(self.signal[0], self.signal[1:] - self.pre_emphasis_katsayisi * self.signal[:-1])
        self.g_mfcc = librosa.feature.mfcc(
            y=self.signal,
            sr=self.TSR,       
            n_mfcc=13,
            n_fft=512,       
            hop_length=256
        )
bulundugu_klasor = os.path.dirname(os.path.abspath(__file__))
veri_dizini = os.path.join(bulundugu_klasor, "Asthma Detection Dataset Version 2") 
kategoriler = ["asthma", "Bronchial", "copd", "healthy", "pneumonia"]
X_veriler = [] # MFCC matrisleri
y_etiketler = [] # Hastalık 

for etiket in kategoriler:
    klasor_yolu = os.path.join(veri_dizini, etiket)
    ## .wav dosyalarını çek.
    wav_dosyalar = [d for d in os.listdir(klasor_yolu) if d.endswith(".wav")]
    toplam_dosya = len(wav_dosyalar)
    
    print(f"--- [{etiket}] klasöründen toplam {toplam_dosya} ses dosyası işlenecek. ---")
    
    ## her wav dosyasının dosya yollarını kopyalarak class içerisindeki fonksiyonda kullan. Enumerate ile numaralandır.
    for i, dosya_adi in enumerate(wav_dosyalar):
        dosya_yolu = os.path.join(klasor_yolu, dosya_adi)

        ses_analiz = Proje(dosya_yolu, TSR=16000)
        mfcc_matrisi = ses_analiz.g_mfcc
        
        X_veriler.append(mfcc_matrisi)
        y_etiketler.append(etiket)
        ## Progress'i görmek amaçlı her 10 file için dosya sayısını yazdır.
        if (i + 1) % 10 == 0 or (i + 1) == toplam_dosya:
            print(f"  -> {etiket}: {i + 1}/{toplam_dosya} dosya tamamlandı...")

print(f"\nBÜYÜK İŞLEM BİTTİ! Toplam {len(X_veriler)} adet ses dosyası başarıyla işlendi.")


X_train, X_test, y_train, y_test = train_test_split(
    X_veriler, 
    y_etiketler, 
    test_size=0.20, 
    random_state=42, 
    stratify=y_etiketler
)

print(f"Eğitim verisi sayısı: {len(X_train)}")
print(f"Test verisi sayısı: {len(X_test)}")
