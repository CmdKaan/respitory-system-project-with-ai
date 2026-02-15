import librosa
import numpy as np
import matplotlib.pyplot as plt

class Proje:
    def __init__(self,sayisal_veri ,TSR=16000):
        self.TSR = TSR
        self.pre_emphasis_katsayisi = 0.97
        self.signal = None
        self.signal = np.array(sayisal_veri, dtype=np.float32)
        """
        Normalizasyon yapiyoruz
        """
        max_val = np.max(np.abs(self.signal))
        if max_val > 0:
            self.signal = self.signal / max_val
        self.signal = np.append(self.signal[0], self.signal[1:] - self.pre_emphasis_katsayisi * self.signal[:-1])
        """
        MFCC verilerini çıkarıyoruz
        """
        self.g_mfcc = librosa.feature.mfcc(
            y=self.signal,
            sr=self.TSR,       
            n_mfcc=13,
            n_fft=512,       
            hop_length=256
        )
    def grafigi_ciz(self):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.g_mfcc, 
                                 x_axis='time', 
                                 sr=self.TSR, 
                                 hop_length=256,
                                 cmap='viridis') 
        plt.colorbar(format='%+2.0f dB') 
        plt.title('MFCC - Akciğer Sesi Analizi')
        plt.ylabel('MFCC Katsayıları')
        plt.xlabel('Zaman (Saniye)')
        plt.tight_layout()
        plt.show()
ornek_veri = np.random.uniform(low=-0.5, high=0.5, size=(16000 * 5)) 
ses_analiz = Proje(ornek_veri,TSR=16000)
ses_analiz.grafigi_ciz()
