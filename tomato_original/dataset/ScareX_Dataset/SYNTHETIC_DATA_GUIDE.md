# ScareX Synthetic Data Guidelines

To improve the robustness of models without extensive field data collection, follow these synthetic generation strategies:

## 1. Image Compositing
- **Foreground:** Segment birds from clear images (using SAM or similar tools).
- **Background:** Collect images of agricultural fields (Paddy, Sugarcane) under various lighting conditions.
- **Blending:** Paste the foreground onto the background. Use Poisson blending or simple alpha compositing with random scaling.
- **Labeling:** Automatically generate bounding boxes based on the pasted foreground mask.

## 2. Audio Mixing
- **Clean Signal:** Use high-quality bird calls.
- **Noise:** Use negative class recordings (Wind, Tractor, Rain).
- **Mixing:** Overlay the clean signal with noise at various SNR (Signal-to-Noise Ratio) levels (e.g., -5dB, 0dB, +5dB).
- **Export:** Export as 16kHz mono `.wav`.

*Note: Always flag synthetic data in the metadata CSVs with `is_synthetic=True` and append `_synth` to the filename.*
