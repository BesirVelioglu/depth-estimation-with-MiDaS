
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import matplotlib.pyplot as plt

# MiDaS modelini yükle
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Giriş dönüştürme pipeline'ını yükle
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Video dosyasını yükle
video_path = 'C:/Staj-projects/CodeThatMidas/kayik.mp4'  # Buraya video dosyanızın yolunu yazın
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Video karesini MiDaS için dönüştür
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Tahmin yap
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

    # Derinlik haritasını göster
    plt.imshow(output, cmap='inferno')
    plt.colorbar()
    plt.title('Depth Map')
    plt.pause(0.00001)
    plt.clf()  # Yeni kare için önceki görüntüyü temizle

    # Orijinal kareyi göster
    cv2.imshow('Original Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close() 