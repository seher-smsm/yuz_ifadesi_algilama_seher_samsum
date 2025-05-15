# yuz_ifadesi_algilama_seher_samsum

# 1- yuz_algila.py (veri toplama) dosyasında, Kamera üzerinden yüz ifadelerini (mutlu, üzgün, kızgın, şaşkın) tanıyıp her bir ifadedeki 468 yüz noktasının X, Y koordinatlarını bir CSV dosyası oluşturduk.

# 2- egitim.py (model eğitimi)

df = pd.read_csv("veriseti.csv")
y = df["Etiket"]
X = df.drop("Etiket", axis=1)
# veriseti.csv dosyasını okur.
# Etiket sütunu hedef değişken (y), geri kalanları girdi özellikleri (X) olur.

pipeline.fit(Xegt, Yegt)
# Eğitim verileriyle pipeline çalıştırılır.
# StandardScaler veriyi dönüştürür.
# LogisticRegression modeli eğitilir.

Y_model = pipeline.predict(Xtst)
dogruluk_orani = accuracy_score(Ytst, Y_model)
print(f"Doğruluk Oranı = {dogruluk_orani}")
# Test verileriyle tahmin yapar.
# Elde edilen doğruluk oranı ekrana yazdırır.

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
    #Eğitilen pipeline (Scaler + Model) model.pkl olarak kaydeder.

# 3-yuz_algila_test.py (gerçek zamanlı test)

draw_landmarks_on_image() 
# fonksiyonu ile Her yüz için landmark noktalarının x ve y koordinatlarını alır,
# Modelle tahmin yapar (model.predict)
# Tahmini (örneğin happy) ve emoji’yi görüntüye yazdırır
# Son olarak bu işlenmiş görüntüyü döndürür.

emoji = emoji_grubu.get(sonuc, "")
# Model sonucuna göre uygun emoji seçer.

annotated_image = cv2.putText(annotated_image, f"{sonuc} {emoji}", ...)
# Görüntüye ifade ve emoji yazar.

cam = cv2.VideoCapture(0)
while cam.isOpened():
...
# Kamera çalışır, yüz tespiti yapar.

cv2.imshow(...): 
# Sonucu  gösterir.

# q tuşuna basılırsa da çıkış yapar.

# 3-veriseti.csv
# Bu dosyada her bir satır, bir yüz görüntüsüne ait landmark koordinatlarını (x ve y değerleri) ve o görüntünün etiketini (mutlu, üzgün, kızgın, şaşkın) içeriyor.


    






