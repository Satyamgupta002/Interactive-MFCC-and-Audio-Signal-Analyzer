import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.fftpack
import soundfile as sf
from datetime import datetime
import uuid

# ========== Helper Functions ==========

def load_audio(file, sr):
    signal, orig_sr = librosa.load(file, sr=sr)
    return signal, sr

def frame_signal(signal, frame_size, frame_stride, sample_rate):
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def apply_hamming(frames):
    return frames * np.hamming(frames.shape[1])

def compute_fft(frames, NFFT):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    return pow_frames

def mel_filter_bank(pow_frames, sample_rate, nfilt, NFFT):
    mel_fbanks = librosa.filters.mel(sr=sample_rate, n_fft=NFFT, n_mels=nfilt)
    mel_energy = np.dot(pow_frames, mel_fbanks.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    return mel_energy, mel_fbanks

def compute_mfcc(mel_energy, num_ceps=13):
    log_mel_energy = np.log(mel_energy)
    mfcc = scipy.fftpack.dct(log_mel_energy, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]
    return mfcc

def plot_waveform_and_spectrum(signal, sr):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(np.arange(len(signal)) / sr, signal)
    axs[0].set_title("Time-Domain Signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1/sr)
    axs[1].plot(freq[:len(freq)//2], np.abs(fft)[:len(freq)//2])
    axs[1].set_title("Frequency-Domain Signal")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    return fig

def plot_spectrogram(signal, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=signal, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    return fig

def plot_mfcc_heatmap(mfcc):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mfcc.T, aspect='auto', origin='lower', interpolation='nearest')
    fig.colorbar(im, ax=ax)
    ax.set_title("MFCC Coefficients Heatmap")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("MFCC Coefficients")
    return fig

def plot_mel_filterbank(mel_fbanks):
    fig, ax = plt.subplots(figsize=(10, 4))
    for m in mel_fbanks:
        ax.plot(m)
    ax.set_title("Mel Filter Bank")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Amplitude")
    return fig

# ========== Sidebar Controls ==========

st.sidebar.markdown("## Configuration")

sample_rate = st.sidebar.selectbox("Sampling Rate (Hz)", [8000, 16000, 22050, 44000], index=1)
frame_size_ms = st.sidebar.slider("Frame Size (ms)", 20, 50, 25)
overlap_percent = st.sidebar.slider("Frame Overlap (%)", 25, 75, 50)
nfilt = st.sidebar.selectbox("Number of Mel Filters", [20, 30, 40, 50, 60], index=0)

# Add timestamp and session ID to sidebar
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
user_id = str(uuid.uuid4())
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Execution Time:** {timestamp}")
st.sidebar.markdown(f"**Session UUID:** `{user_id}`")

# ========== App Header ==========

st.markdown("<h2 style='text-align: center;'>Interactive MFCC and Audio Signal Analyzer</h2>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center;'>Submitted By: Satyam Gupta</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Scholar No.: 2311401167</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Audio File Used: s7.wav</h4>", unsafe_allow_html=True)

# ========== Load Audio ==========
uploaded_file = st.file_uploader("s7.wav", type="wav")


# ========== Tabbed Section Layout ==========

if uploaded_file:
    signal, sr = load_audio(uploaded_file, sample_rate)
    st.audio(uploaded_file)
    tab1, tab2, tab3, tab4 = st.tabs([
    "1. Time & Frequency Domain Analysis",
    "2. Spectrogram vs. MFCC Comparison",
    "3. Interactive Frame and Overlap Configuration",
    "4. Mel Filter Bank Customization"
])

    with tab1:
        st.subheader("1. Time & Frequency Domain Analysis")
        st.pyplot(plot_waveform_and_spectrum(signal, sr))
        st.markdown("""
        - The time-domain plot shows amplitude variations over time.
        - It helps identify silence, speech, or noise regions.
        - Frequency domain reveals dominant frequencies in the signal.
        - Used for identifying harmonics, pitch, and timbre.
        - Useful in detecting periodic patterns or noise.
        """)

    with tab2:
        st.subheader("2. Spectrogram vs. MFCC Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_spectrogram(signal, sr))
            st.markdown("""
            - Spectrogram shows how frequency content changes over time.
            - Mel scale mimics human auditory perception.
            - Bright regions indicate high energy at those frequencies.
            - Helps distinguish vowels, consonants, and intonation.
            - Ideal for speech/music analysis.
            """)
        with col2:
            frames = frame_signal(signal, 0.025, 0.01, sr)
            frames = apply_hamming(frames)
            pow_frames = compute_fft(frames, 512)
            mel_energy, _ = mel_filter_bank(pow_frames, sr, 20, 512)
            mfcc = compute_mfcc(mel_energy)
            st.pyplot(plot_mfcc_heatmap(mfcc))
            st.markdown("""
            - MFCCs represent short-term power spectrum of sound.
            - Compact representation ideal for machine learning.
            - Derived from log of Mel spectrogram.
            - Used in speech, music, and emotion recognition.
            - Each coefficient reflects energy in a Mel band.
            """)

    with tab3:
        st.subheader("3. Interactive Frame and Overlap Configuration")
        frame_size = frame_size_ms / 1000.0
        frame_stride = frame_size * (1 - overlap_percent / 100.0)
        frames = frame_signal(signal, frame_size, frame_stride, sr)
        frames = apply_hamming(frames)
        pow_frames = compute_fft(frames, 512)
        mel_energy, _ = mel_filter_bank(pow_frames, sr, 20, 512)
        mfcc = compute_mfcc(mel_energy)
        st.pyplot(plot_mfcc_heatmap(mfcc))
        st.markdown("""
        - Adjusting frame size and overlap alters time-frequency resolution.
        - Smaller frames capture fine changes; large frames give smooth estimates.
        - Overlap prevents data loss and improves continuity.
        - Useful to understand feature extraction sensitivity.
        - Helps optimize MFCCs for specific tasks.
        """)

    with tab4:
        st.subheader("4. Mel Filter Bank Customization")
        frames = frame_signal(signal, 0.025, 0.01, sr)
        frames = apply_hamming(frames)
        pow_frames = compute_fft(frames, 512)
        mel_energy, mel_fbanks = mel_filter_bank(pow_frames, sr, nfilt, 512)
        mfcc = compute_mfcc(mel_energy)
        st.pyplot(plot_mel_filterbank(mel_fbanks))
        st.markdown("""
        - Shows filter responses across frequency bands.
        - More filters improve resolution at cost of complexity.
        - Filters are triangular and overlap.
        - Designed to match human ear perception.
        - Essential step in MFCC calculation.
        """)
        st.pyplot(plot_mfcc_heatmap(mfcc))
        st.markdown("""
        - MFCCs after customizing filter banks.
        - Demonstrates how filter count affects features.
        - Useful for tuning performance in classification tasks.
        """)


