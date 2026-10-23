# Grayscale input decision

Observed image modes: {'L': 1800}

The images carry a single intensity channel (stored as 'L', or as RGB with identical channels).

## Recommendation
* **Custom CNN (trained from scratch): use a single-channel (1x H x W) input.** All colour
  channels would be identical copies, so 3 channels add no information, triple the first-layer
  weights and the memory for the input batch. Normalise with the dataset's own mean/std (computed
  on the TRAINING split only).
* **Pretrained ImageNet models (ResNet18, EfficientNet-B0): feed 3 channels** by replicating the gray
  channel (or, equivalently, sum conv1 weights over the RGB axis to make a 1-channel stem). Replication is
  simplest and keeps the pretrained filters exactly as trained; normalise with ImageNet mean/std
  because that is the input distribution those weights expect.
* Both choices are implemented later in preprocessing.py so the comparison stays fair.
