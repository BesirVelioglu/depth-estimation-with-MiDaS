# Depth Estimation with MiDaS

This project demonstrates the use of the MiDaS model for depth estimation from a single image. The MiDaS model is designed to generate depth maps that can be useful in various applications such as scene understanding, AR, and robotics.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Matplotlib

To install the required packages, run:

```bash
pip install torch opencv-python matplotlib

```
Load the MiDaS Model
The MiDaS model is loaded directly from the PyTorch Hub. In this project, we use the MiDaS_small version for CPU-based operations.

```bash
import torch

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
```

