## Interactive MFCC and Audio Signal Analyzer

An interactive audio signal analysis tool built using **Streamlit** to visualize waveform, frequency and time spectrum, spectrogram, and MFCC(Mel-Frequency Cepstral Coefficients) features. The application allows users to explore how different signal processing parameters affect feature extraction.

---

## Features

* **Time & Frequency Domain Analysis**

  * Visualize waveform (time-domain)
  * Analyze frequency spectrum using FFT

* **Spectrogram vs MFCC Comparison**

  * Mel Spectrogram visualization
  * MFCC heatmap representation
  * Understand differences between spectral and cepstral features

* **Interactive Frame & Overlap Configuration**

  * Adjust frame size and overlap
  * Observe impact on MFCC extraction

* **Mel Filter Bank Customization**

  * Configure number of Mel filters (20–60)
  * Visualize filter banks
  * Analyze effect on MFCC features

* **Audio Upload Support**

  * Upload `.wav` files and analyze instantly

---

## Tech Stack

* **Frontend/UI**: Streamlit
* **Audio Processing**: Librosa, SciPy
* **Visualization**: Matplotlib
* **Numerical Computing**: NumPy

---

## Concepts Covered

* Signal Framing & Windowing (Hamming Window)
* Fast Fourier Transform (FFT)
* Power Spectrum Analysis
* Mel Filter Banks
* MFCC (Mel-Frequency Cepstral Coefficients)
* Spectrogram Analysis

---

## How It Works

1. Upload an audio `.wav` file
2. Select parameters:

   * Sampling rate
   * Frame size & overlap
   * Number of Mel filters
3. Explore:

   * Waveform & frequency spectrum
   * Spectrogram vs MFCC
   * Effect of parameter changes on feature extraction

---

## Getting Started

```bash
git clone https://github.com/Satyamgupta002/Interactive-MFCC-and-Audio-Signal-Analyzer.git
pip install -r requirements.txt
streamlit run app_2311401167.py
```
---
## Live Link

#### <a>https://interactive-mfcc-and-audio-signal-analyzer-satyam-gupta.streamlit.app/</a>
---
## Use Cases

* Audio signal processing learning
* Speech & music analysis
* Understanding MFCC for ML models
* Feature engineering experimentation

---

## Author

**Satyam Gupta**

B.Tech, Electronics & Communication Engineering <br>
MANIT Bhopal

---

## Show Your Support

If you like this project, consider giving it a ⭐ on GitHub!
