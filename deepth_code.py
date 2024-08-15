# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:34:32 2024

@author: mbesir
"""

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

# Fotoğrafı yükle
image_path = 'C:/CodeThatMidas/dolap.jpg'  # Buraya fotoğrafınızın yolunu yazın
frame = cv2.imread(image_path)

# Fotoğrafın başarıyla yüklenip yüklenmediğini kontrol edin
if frame is None:
    print(f"Error: Could not read image file {image_path}. Please check the file path and integrity.")
else:
    # Fotoğrafı MiDaS için dönüştür
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
    plt.show()

    # Orijinal fotoğrafı göster
    cv2.imshow('Original Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
