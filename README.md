
# Image Similarity metrics

## Setup
```bash
python download_models.py
```

```python
from criteria.id_loss import IDLoss
from criteria.lpips.lpips import LPIPS

lpips =  LPIPS(net_type='vgg').to(device)
arcface = IDLoss().to(device).eval()

# extract features.
# x = your image 
lpips.net(x)
arcface.extract_feats(x)

```