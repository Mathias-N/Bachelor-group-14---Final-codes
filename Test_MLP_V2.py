import logging
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtGui import QFont
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from scipy.signal import welch, butter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter1d
from scipy.stats import kurtosis, entropy
from scipy.fft import fft, ifft
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from imblearn.over_sampling import SMOTE
import pickle
import os

# Globale konstanter
N_PERSEG = 256
N_OVERLAP = 128
MAX_CONCURRENT_THREADS = 5

# Sjekk scikit-learn-versjon
import sklearn
print(f"scikit-learn versjon: {sklearn.__version__}")

# Signalbehandlingsfunksjoner
def notch_filter(signal, fs, freq=50, q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, signal)

def highpass_filter(signal, fs, low_freq=20, order=3):
    nyquist = fs / 2
    low = low_freq / nyquist
    b, a = butter(order, low, btype='high')
    return filtfilt(b, a, signal)

def bandpass_filter(signal, fs, low_freq=30, high_freq=100, gain=10.0):
    nyquist = fs / 2
    low = low_freq / nyquist
    high = min(high_freq, nyquist - 1e-6) / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal - np.mean(signal))
    return filtered * gain

def fft_noise_removal(signal, fs, low_freq=20, high_freq=150):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_signal = fft(signal)
    mask = (np.abs(freqs) < low_freq) | (np.abs(freqs) > high_freq)
    fft_signal[mask] = 0
    cleaned_signal = ifft(fft_signal).real
    return cleaned_signal

def hjorth_parameters(signal):
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    var_signal = np.var(signal)
    var_diff1 = np.var(diff1)
    var_diff2 = np.var(diff2)
    mobility = np.sqrt(var_diff1 / var_signal) if var_signal > 0 else 0
    complexity = (np.sqrt(var_diff2 / var_diff1) / mobility) if var_diff1 > 0 and mobility > 0 else 0
    return mobility, complexity

def lms_filter(signal, reference, mu=0.1, filter_length=20):
    n_samples = len(signal)
    w = np.zeros(filter_length)
    y = np.zeros(n_samples)
    e = np.zeros(n_samples)
    for n in range(filter_length, n_samples):
        x_n = reference[n-filter_length:n][::-1]
        y[n] = np.dot(w, x_n)
        e[n] = signal[n] - y[n]
        w += 2 * mu * e[n] * x_n
    return e

def process_signal_for_features(signal, fs, imu_reference=None):
    signal_highpassed = highpass_filter(signal, fs, low_freq=20)
    if imu_reference is not None:
        signal_cleaned = lms_filter(signal_highpassed, imu_reference, mu=0.01, filter_length=10)
    else:
        signal_cleaned = signal_highpassed
    signal_notched = notch_filter(signal_cleaned, fs, freq=50)
    signal_fft_cleaned = fft_noise_removal(signal_notched, fs, low_freq=20, high_freq=150)
    signal_rectified = np.abs(signal_fft_cleaned)
    filtered_signal = bandpass_filter(signal_rectified, fs)
    filtered_signal = uniform_filter1d(filtered_signal, size=5)
    return filtered_signal

def extract_features(signal, fs, imu_reference=None):
    filtered_signal = process_signal_for_features(signal, fs, imu_reference=imu_reference)
    rms = np.sqrt(np.mean(filtered_signal**2))
    slope = np.mean(np.diff(filtered_signal))
    f, Pxx = welch(filtered_signal, fs=fs, nperseg=min(N_PERSEG, len(signal)), noverlap=min(N_OVERLAP, len(signal)//2))
    spec_energy_20_150 = np.sum(Pxx[(f >= 20) & (f <= 150)]) if any((f >= 20) & (f <= 150)) else 0
    median_freq = f[np.where(np.cumsum(Pxx) >= np.sum(Pxx)/2)[0][0]]
    spec_energy_20_70 = np.sum(Pxx[(f >= 20) & (f <= 70)]) if any((f >= 20) & (f <= 70)) else 0
    spec_energy_70_150 = np.sum(Pxx[(f > 70) & (f <= 150)]) if any((f > 70) & (f <= 150)) else 0
    freq_ratio = spec_energy_20_70 / spec_energy_70_150 if spec_energy_70_150 > 0 else 0
    zero_crossings = np.sum(np.abs(np.sign(filtered_signal[1:]) - np.sign(filtered_signal[:-1])) > 0) / 2
    zero_crossings = zero_crossings / len(filtered_signal)
    spectral_entropy = entropy(Pxx / np.sum(Pxx)) if np.sum(Pxx) > 0 else 0
    kurt = kurtosis(filtered_signal)
    mobility, complexity = hjorth_parameters(filtered_signal)
    features = [rms, spec_energy_20_150, slope, median_freq, zero_crossings, spectral_entropy, kurt, mobility, complexity, freq_ratio]
    return features

def extract_features_all_channels(data, channels, fs, imu_reference=None):
    all_features = []
    for ch in channels:
        features = extract_features(data[ch], fs, imu_reference=imu_reference)
        all_features.extend(features)
    return np.array(all_features)

# NeuralNetworkClassifier
class NeuralNetworkClassifier:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(40, 20, 10),
            solver='adam',
            learning_rate_init=0.001,
            alpha=0.4,
            max_iter=1000,
            random_state=42
        )
        self.trained = False
        self.num_features_per_channel = 10
        self.num_channels = None
        self.features = []
        self.labels = []

    def compute_class_distribution(self, labels):
        label_counts = Counter(labels)
        print(f"Klassefordeling før balansering: {label_counts}")
        return label_counts

    def balance_classes(self, features, labels):
        if len(set(labels)) <= 1:
            print("Kun én klasse til stede, hopper over SMOTE-balansering.")
            return features, labels
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(features, labels)
        print(f"Klassefordeling etter SMOTE: {Counter(y_balanced)}")
        return X_balanced, y_balanced

    def train(self, features, labels, num_channels):
        self.num_channels = num_channels
        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        self.compute_class_distribution(labels)
        X_balanced, y_balanced = self.balance_classes(X_scaled, labels)
        self.model.fit(X_balanced, y_balanced)
        self.trained = True
        print(f"Modell trent med {len(X_balanced)} balanserte samples")

    def classify(self, features):
        if not self.trained:
            return "Modell ikke trent"
        expected_feature_length = self.num_channels * self.num_features_per_channel
        if len(features) != expected_feature_length:
            raise ValueError(f"Forventet {expected_feature_length} trekk, men fikk {len(features)} trekk.")
        features_scaled = self.scaler.transform([features])
        probs = self.model.predict_proba(features_scaled)[0]
        max_prob = np.max(probs)
        pred = self.model.predict(features_scaled)[0]
        rms = features[0]
        return pred

    def save_model(self, model_file="mlp_model.pkl"):
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "num_channels": self.num_channels,
            "features": self.features,
            "labels": self.labels
        }
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Modell lagret til {model_file}")

    def load_model(self, model_file="mlp_model.pkl"):
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)
                self.model = model_data["model"]
                self.scaler = model_data["scaler"]
                self.num_channels = model_data["num_channels"]
                self.features = model_data.get("features", [])
                self.labels = model_data.get("labels", [])
                self.trained = True
            print(f"Lastet modell fra {model_file}")
            return True
        return False

# SensorData
class SensorData(QtCore.QObject):
    data_updated = QtCore.pyqtSignal(np.ndarray)
    connection_lost = QtCore.pyqtSignal(str)

    def __init__(self, board_shim):
        super().__init__()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(board_shim.get_board_id())
        self.accel_channels = BoardShim.get_accel_channels(board_shim.get_board_id())
        self.gyro_channels = BoardShim.get_gyro_channels(board_shim.get_board_id())
        self.sampling_rate = BoardShim.get_sampling_rate(board_shim.get_board_id())
        self.num_points = 2 * self.sampling_rate
        self.alpha = 0.2
        self.no_data_count = 0
        self.max_no_data_attempts = 10

        self.raw_window = pg.GraphicsLayoutWidget(title="Råsignaler")
        self.rect_window = pg.GraphicsLayoutWidget(title="Rektifiserte signaler")
        self.ema_window = pg.GraphicsLayoutWidget(title="EMA-signaler")
        self.processed_window = pg.GraphicsLayoutWidget(title="Behandlede signaler (input til modell)")

        self.raw_window.resize(1400, 800)
        self.rect_window.resize(1400, 800)
        self.ema_window.resize(1400, 800)
        self.processed_window.resize(1400, 800)

        self.raw_window.show()
        self.rect_window.show()
        self.ema_window.show()
        self.processed_window.show()

        self.raw_plots = []
        self.rect_plots = []
        self.ema_plots = []
        self.processed_plots = []

        raw_layout = self.raw_window.ci.layout
        rect_layout = self.rect_window.ci.layout
        ema_layout = self.ema_window.ci.layout
        processed_layout = self.processed_window.ci.layout

        raw_y_min, raw_y_max = -500, 500
        rect_y_min, rect_y_max = 0, 500
        ema_y_min, ema_y_max = 0, 500
        processed_y_min, processed_y_max = 0, 500

        for i in range(len(self.exg_channels)):
            row = i // 4
            col = i % 4
            raw_plot_item = pg.PlotItem(title=f"Kanal {i+1} Rå")
            raw_plot_item.setYRange(raw_y_min, raw_y_max)
            raw_plot_item.getAxis('bottom').setTickFont(pg.QtGui.QFont('Arial', 12))
            raw_plot_item.getAxis('left').setTickFont(pg.QtGui.QFont('Arial', 12))
            raw_plot_item.setTitle(f"Kanal {i+1} Rå", size='14pt')
            raw_layout.addItem(raw_plot_item, row, col)
            self.raw_plots.append(raw_plot_item.plot(pen='b'))

            rect_plot_item = pg.PlotItem(title=f"Kanal {i+1} Rektifisert")
            rect_plot_item.setYRange(rect_y_min, rect_y_max)
            rect_plot_item.getAxis('bottom').setTickFont(pg.QtGui.QFont('Arial', 12))
            rect_plot_item.getAxis('left').setTickFont(pg.QtGui.QFont('Arial', 12))
            rect_plot_item.setTitle(f"Kanal {i+1} Rektifisert", size='14pt')
            rect_layout.addItem(rect_plot_item, row, col)
            self.rect_plots.append(rect_plot_item.plot(pen='r'))

            ema_plot_item = pg.PlotItem(title=f"Kanal {i+1} EMA")
            ema_plot_item.setYRange(ema_y_min, ema_y_max)
            ema_plot_item.getAxis('bottom').setTickFont(pg.QtGui.QFont('Arial', 12))
            ema_plot_item.getAxis('left').setTickFont(pg.QtGui.QFont('Arial', 12))
            ema_plot_item.setTitle(f"Kanal {i+1} EMA", size='14pt')
            ema_layout.addItem(ema_plot_item, row, col)
            self.ema_plots.append(ema_plot_item.plot(pen='y'))

            processed_plot_item = pg.PlotItem(title=f"Kanal {i+1} Behandlet")
            processed_plot_item.setYRange(processed_y_min, processed_y_max)
            processed_plot_item.getAxis('bottom').setTickFont(pg.QtGui.QFont('Arial', 12))
            processed_plot_item.getAxis('left').setTickFont(pg.QtGui.QFont('Arial', 12))
            processed_plot_item.setTitle(f"Kanal {i+1} Behandlet", size='14pt')
            processed_layout.addItem(processed_plot_item, row, col)
            self.processed_plots.append(processed_plot_item.plot(pen='g'))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def calculate_ema(self, signal):
        ema = np.zeros_like(signal)
        ema[0] = signal[0]
        for i in range(1, len(signal)):
            ema[i] = self.alpha * signal[i] + (1 - self.alpha) * ema[i - 1]
        return ema

    def update(self):
        try:
            data = self.board_shim.get_current_board_data(self.num_points)
            if data is None or data.size == 0:
                self.no_data_count += 1
                print(f"Ingen data mottatt fra board_shim (forsøk {self.no_data_count}/{self.max_no_data_attempts})")
                if self.no_data_count >= self.max_no_data_attempts:
                    self.connection_lost.emit("Mistet forbindelse til MindRove-enheten etter flere forsøk.")
                    self.timer.stop()
                return
            self.no_data_count = 0
            exg_data = np.array([data[ch] for ch in self.exg_channels])
            accel_data = np.array([data[ch] for ch in self.accel_channels]) if self.accel_channels else None
            gyro_data = np.array([data[ch] for ch in self.gyro_channels]) if self.gyro_channels else None
            imu_reference = np.sqrt(np.sum(accel_data**2, axis=0)) if accel_data is not None else None

            for i, channel in enumerate(self.exg_channels):
                raw_signal = data[channel]
                raw_signal_centered = raw_signal - np.mean(raw_signal)
                rectified_signal = np.abs(raw_signal_centered)
                ema_signal = self.calculate_ema(rectified_signal)
                processed_signal = process_signal_for_features(raw_signal, self.sampling_rate, imu_reference=imu_reference)
                self.raw_plots[i].setData(raw_signal_centered)
                self.rect_plots[i].setData(rectified_signal)
                self.ema_plots[i].setData(ema_signal)
                self.processed_plots[i].setData(processed_signal)

            self.data_updated.emit(exg_data)
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            print(f"Feil under oppdatering: {e}")
            self.no_data_count += 1
            if self.no_data_count >= self.max_no_data_attempts:
                self.connection_lost.emit(f"Mistet forbindelse til MindRove-enheten: {str(e)}")
                self.timer.stop()

# FeatureExtractionThread
class FeatureExtractionThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(list, list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, segment, channels, fs, movement_label):
        super().__init__()
        self.segment = segment
        self.channels = channels
        self.fs = fs
        self.movement_label = movement_label
        self.is_running = True

    def run(self):
        try:
            if not self.is_running:
                return
            features_list = []
            labels_list = []
            num_augmentations = 1 if self.movement_label in ["Hånd stille", "Tommel"] else 0
            for _ in range(num_augmentations + 1):
                if _ > 0:
                    signal_copy = {}
                    for ch in self.channels:
                        signal = self.segment[ch].copy()
                        noise = np.random.normal(0, 0.01 * np.std(signal), len(signal))
                        signal += noise
                        scale = np.random.uniform(0.9, 1.1)
                        signal *= scale
                        signal_copy[ch] = signal
                    features = extract_features_all_channels(signal_copy, self.channels, self.fs)
                else:
                    features = extract_features_all_channels(self.segment, self.channels, self.fs)
                features_list.append(features.tolist())
                labels_list.append(self.movement_label)
            for feats, lab in zip(features_list, labels_list):
                self.finished.emit(feats, [lab])
        except Exception as e:
            error_msg = f"Feil i FeatureExtractionThread for {self.movement_label}: {str(e)}"
            print(error_msg)
            self.error.emit(error_msg)
            self.finished.emit([0] * (10 * len(self.channels)), [self.movement_label])
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False
        self.terminate()
        self.wait()

# CalibrationWindow
class CalibrationWindow(QtWidgets.QWidget):
    def __init__(self, classifier, sensor_data):
        super().__init__()
        self.classifier = classifier
        self.sensor_data = sensor_data
        self.features = []
        self.labels = []
        self.calibration_counts = {}
        self.time_remaining = 20
        self.window_size = self.sensor_data.num_points
        self.step_size = self.window_size // 2
        self.current_label = None
        self.is_calibrating = False
        self.calibration_buffer = []
        self.active_threads = []

        self.setWindowTitle("Kalibrering")
        self.resize(800, 600)
        self.layout = QtWidgets.QVBoxLayout()

        self.info_label = QtWidgets.QLabel("Trykk 'Start Kalibrering' for å begynne")
        self.info_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.layout.addWidget(self.info_label)

        self.calibrate_button = QtWidgets.QPushButton("Start Kalibrering")
        self.calibrate_button.setStyleSheet("font-size: 14pt;")
        self.calibrate_button.clicked.connect(self.start_calibrering)
        self.layout.addWidget(self.calibrate_button)

        self.test_button = QtWidgets.QPushButton("Start Testing")
        self.test_button.setStyleSheet("font-size: 14pt;")
        self.test_button.setEnabled(False)
        self.test_button.clicked.connect(self.start_testing)
        self.layout.addWidget(self.test_button)

        self.corr_button = QtWidgets.QPushButton("Plott Korrelasjonsmatrise")
        self.corr_button.setStyleSheet("font-size: 14pt;")
        self.corr_button.setEnabled(False)
        self.corr_button.clicked.connect(self.plot_correlation)
        self.layout.addWidget(self.corr_button)

        self.plot_features_button = QtWidgets.QPushButton("Vis Trekk")
        self.plot_features_button.setStyleSheet("font-size: 14pt;")
        self.plot_features_button.setEnabled(False)
        self.plot_features_button.clicked.connect(self.plot_features)
        self.layout.addWidget(self.plot_features_button)

        self.setLayout(self.layout)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.sensor_data.data_updated.connect(self.record_data)

    def start_calibrering(self):
        if self.active_threads:
            self.stop_all_threads()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Kalibrering")
        dialog.resize(600, 400)
        dialog_layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel("Velg bevegelsesnummer (0=Hånd stille, 1=Tommel, 2=Pekefinger, 3=Langfinger, 4=Ringfinger, 5=Lillefinger, 6=Knyttneve):")
        label.setStyleSheet("font-size: 14pt;")
        dialog_layout.addWidget(label)
        spin_box = QtWidgets.QSpinBox()
        spin_box.setRange(0, 6)
        spin_box.setFont(QFont("Arial", 14))
        dialog_layout.addWidget(spin_box)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.setStyleSheet("font-size: 14pt;")
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)
        dialog.setLayout(dialog_layout)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            movement_num = spin_box.value()
            movement_labels = {0: "Hånd stille", 1: "Tommel", 2: "Pekefinger", 3: "Langfinger", 4: "Ringfinger", 5: "Lillefinger", 6: "Knyttneve"}
            self.current_label = movement_labels[movement_num]
            self.start_time = time.time()
            self.is_calibrating = True
            self.calibration_buffer = []
            self.info_label.setText(f"Kalibrerer: {self.current_label} (Bevegelse {movement_num}) ({self.time_remaining}s igjen)")
            self.timer.start(500)

    def update_timer(self):
        elapsed_time = time.time() - self.start_time
        self.time_remaining = max(0, 20 - int(elapsed_time))
        movement_labels = {0: "Hånd stille", 1: "Tommel", 2: "Pekefinger", 3: "Langfinger", 4: "Ringfinger", 5: "Lillefinger", 6: "Knyttneve"}
        movement_map = {v: k for k, v in movement_labels.items()}
        movement_num = movement_map[self.current_label]
        self.info_label.setText(f"Kalibrerer: {self.current_label} (Bevegelse {movement_num}) ({self.time_remaining}s igjen)")
        if self.time_remaining <= 0:
            self.timer.stop()
            self.stop_calibrering()

    def record_data(self, exg_data):
        if self.is_calibrating:
            self.calibration_buffer.append(exg_data)

    def stop_calibrering(self):
        self.is_calibrating = False
        if len(self.calibration_buffer) > 0:
            channel_data = np.concatenate(self.calibration_buffer, axis=1)
            num_samples = (channel_data.shape[1] - self.window_size) // self.step_size + 1
            for i in range(max(1, num_samples)):
                while len([t for t in self.active_threads if t.isRunning()]) >= MAX_CONCURRENT_THREADS:
                    QtWidgets.QApplication.processEvents()
                    time.sleep(0.1)
                start = i * self.step_size
                end = start + self.window_size
                if end <= channel_data.shape[1]:
                    segment = channel_data[:, start:end]
                    thread = FeatureExtractionThread(segment, self.sensor_data.exg_channels, self.sensor_data.sampling_rate, self.current_label)
                    thread.finished.connect(lambda feats, labs: self.handle_features(feats, labs))
                    thread.error.connect(lambda error_msg: print(f"Trådfeil: {error_msg}"))
                    thread.start()
                    self.active_threads.append(thread)
            print(f"Genererte {num_samples} samples for {self.current_label} i denne kalibreringen")
        else:
            self.info_label.setText(f"Kalibrering av {self.current_label} fullført! (0 ganger)")

    def handle_features(self, features, labels):
        self.features.append(features)
        self.labels.append(labels[0])
        self.classifier.features.append(features)
        self.classifier.labels.append(labels[0])
        self.calibration_counts[labels[0]] = self.calibration_counts.get(labels[0], 0) + 1
        count = self.calibration_counts[labels[0]]
        movement_labels = {0: "Hånd stille", 1: "Tommel", 2: "Pekefinger", 3: "Langfinger", 4: "Ringfinger", 5: "Lillefinger", 6: "Knyttneve"}
        movement_num = {v: k for k, v in movement_labels.items()}[labels[0]]
        self.info_label.setText(f"Kalibrering av {labels[0]} (Bevegelse {movement_num}) fullført! ({count} samples)")
        required_movements = set(movement_labels.values())
        if set(self.calibration_counts.keys()) == required_movements and all(c >= 2 for c in self.calibration_counts.values()):
            self.test_button.setEnabled(True)
            self.corr_button.setEnabled(True)
            self.plot_features_button.setEnabled(True)

    def stop_all_threads(self):
        for thread in self.active_threads:
            if thread.isRunning():
                thread.stop()
        for thread in self.active_threads:
            if thread.isRunning():
                thread.wait()
        self.active_threads.clear()

    def closeEvent(self, event):
        self.stop_all_threads()
        event.accept()

    def start_testing(self):
        self.classifier.train(self.features, self.labels, num_channels=len(self.sensor_data.exg_channels))
        self.classifier.features = self.features
        self.classifier.labels = self.labels
        self.test_window = TestingWindow(self.classifier, self.sensor_data, self)
        self.test_window.show()

    def plot_correlation(self):
        features_array = np.array(self.features)
        corr_matrix = np.corrcoef(features_array, rowvar=False)
        num_channels = len(self.sensor_data.exg_channels)
        feature_labels = ["RMS", "SPEC_ENERGY_20_150", "Slope", "MedianFreq", "ZeroCrossings", "SpectralEntropy", "Kurtosis", "Mobility", "Complexity", "FreqRatio"]
        selected_names = [f"Ch{ch+1}_{label}" for ch in range(num_channels) for label in feature_labels]
        df_corr = pd.DataFrame(corr_matrix, index=selected_names, columns=selected_names)
        plt.figure(figsize=(30, 28))
        sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Korrelasjonsmatrise for trekk")
        plt.tight_layout()
        plt.show()

    def plot_features(self):
        features_array = np.array(self.features)
        labels_array = np.array(self.labels)
        plt.figure(figsize=(12, 8))
        for movement in set(self.labels):
            mask = labels_array == movement
            plt.scatter(features_array[mask, 0], features_array[mask, 2], label=movement, alpha=0.5)
        plt.xlabel("RMS")
        plt.ylabel("Slope")
        plt.legend()
        plt.title("Spredningsplott av trekk (RMS vs Slope)")
        plt.show()

# TestingWindow
class TestingWindow(QtWidgets.QWidget):
    def __init__(self, classifier, sensor_data, calibration_window):
        super().__init__()
        self.classifier = classifier
        self.sensor_data = sensor_data
        self.calibration_window = calibration_window
        self.setWindowTitle("Live Testing og Justering")
        self.resize(600, 400)
        self.layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel("Hånd stille")
        self.label.setStyleSheet("font-size: 24pt; color: blue; font-weight: bold; text-align: center;")
        self.layout.addWidget(self.label)

        self.eval_button = QtWidgets.QPushButton("Start Evaluering og Justering")
        self.eval_button.setStyleSheet("font-size: 14pt;")
        self.eval_button.clicked.connect(self.start_evaluation)
        self.layout.addWidget(self.eval_button)

        self.save_button = QtWidgets.QPushButton("Lagre Modell")
        self.save_button.setStyleSheet("font-size: 14pt;")
        self.save_button.clicked.connect(self.save_model)
        self.layout.addWidget(self.save_button)

        self.save_data_button = QtWidgets.QPushButton("Lagre Nye Data")
        self.save_data_button.setStyleSheet("font-size: 14pt;")
        self.save_data_button.setEnabled(False)
        self.save_data_button.clicked.connect(self.save_new_data)
        self.layout.addWidget(self.save_data_button)

        self.eval_result_label = QtWidgets.QLabel("")
        self.eval_result_label.setStyleSheet("font-size: 14pt;")
        self.layout.addWidget(self.eval_result_label)

        self.setLayout(self.layout)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_test)
        self.timer.start(500)

        self.eval_mode = False
        self.eval_movement = None
        self.eval_time_remaining = 0
        self.eval_predictions = []
        self.eval_buffer = []
        self.new_features = []
        self.new_labels = []

    def start_evaluation(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Velg Bevegelse å Evaluere")
        dialog.resize(800, 500)
        dialog_layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel("Velg bevegelse å teste (30 sekunder):")
        label.setStyleSheet("font-size: 18pt;")
        dialog_layout.addWidget(label)
        combo = QtWidgets.QComboBox()
        movements = ["Hånd stille", "Tommel", "Pekefinger", "Langfinger", "Ringfinger", "Lillefinger", "Knyttneve"]
        combo.addItems(movements)
        combo.setStyleSheet("font-size: 16pt;")
        dialog_layout.addWidget(combo)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.setStyleSheet("font-size: 16pt;")
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)
        dialog.setLayout(dialog_layout)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.eval_movement = combo.currentText()
            self.eval_time_remaining = 30
            self.eval_mode = True
            self.eval_predictions = []
            self.eval_buffer = []
            self.new_features = []
            self.new_labels = []
            self.label.setText(f"Evaluerer: {self.eval_movement} ({self.eval_time_remaining}s igjen)")
            self.eval_timer = QtCore.QTimer()
            self.eval_timer.timeout.connect(self.update_eval_timer)
            self.eval_timer.start(1000)

    def update_eval_timer(self):
        self.eval_time_remaining -= 1
        if self.eval_time_remaining <= 0:
            self.eval_timer.stop()
            self.finish_evaluation()
        else:
            self.label.setText(f"Evaluerer: {self.eval_movement} ({self.eval_time_remaining}s igjen)")

    def update_test(self):
        try:
            data = self.sensor_data.board_shim.get_current_board_data(self.sensor_data.num_points)
            window_size = self.sensor_data.num_points
            if data.shape[1] >= window_size:
                latest_data = data[:, -window_size:]
                exg_data = np.array([latest_data[ch] for ch in self.sensor_data.exg_channels])
                accel_data = np.array([latest_data[ch] for ch in self.sensor_data.accel_channels]) if self.sensor_data.accel_channels else None
                imu_reference = np.sqrt(np.sum(accel_data**2, axis=0)) if accel_data is not None else None
                features = extract_features_all_channels(exg_data, self.sensor_data.exg_channels, self.sensor_data.sampling_rate, imu_reference=imu_reference)
                predicted_movement = self.classifier.classify(features)
                if self.eval_mode:
                    self.eval_predictions.append(predicted_movement)
                    self.eval_buffer.append(exg_data)
                    self.new_features.append(features)
                    self.new_labels.append(self.eval_movement)
                else:
                    self.label.setText(predicted_movement)
            else:
                if not self.eval_mode:
                    self.label.setText("Hånd stille")
        except Exception as e:
            print(f"Feil under testing: {e}")
            self.label.setText("Feil")

    def finish_evaluation(self):
        self.eval_mode = False
        correct_preds = sum(1 for pred in self.eval_predictions if pred == self.eval_movement)
        total_preds = len(self.eval_predictions)
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        self.eval_result_label.setText(f"Nøyaktighet for {self.eval_movement}: {accuracy:.2%} ({correct_preds}/{total_preds})")
        self.save_data_button.setEnabled(True)
        if accuracy < 0.85:
            print(f"Nøyaktighet lav ({accuracy:.2%}), justerer modell for {self.eval_movement}")
            self.adjust_model()
            self.eval_result_label.setText(f"Modell justert for {self.eval_movement}. Test på nytt om ønskelig.")
        else:
            print(f"Nøyaktighet akseptabel ({accuracy:.2%}), ingen justering nødvendig.")

    def adjust_model(self):
        channel_data = np.concatenate(self.eval_buffer, axis=1)
        num_samples = (channel_data.shape[1] - self.sensor_data.num_points) // (self.sensor_data.num_points // 2) + 1
        base_features = self.classifier.features.copy() if self.classifier.features else []
        base_labels = self.classifier.labels.copy() if self.classifier.labels else []
        new_features = base_features[:]
        new_labels = base_labels[:]
        for i in range(max(1, num_samples)):
            start = i * (self.sensor_data.num_points // 2)
            end = start + self.sensor_data.num_points
            if end <= channel_data.shape[1]:
                segment = channel_data[:, start:end]
                features = extract_features_all_channels(segment, self.sensor_data.exg_channels, self.sensor_data.sampling_rate)
                new_features.append(features)
                new_labels.append(self.eval_movement)
        self.classifier.features = new_features
        self.classifier.labels = new_labels
        self.calibration_window.features = new_features
        self.calibration_window.labels = new_labels
        self.classifier.train(new_features, new_labels, num_channels=len(self.sensor_data.exg_channels))

    def save_model(self):
        self.classifier.save_model()
        self.eval_result_label.setText("Modell lagret!")

    def save_new_data(self):
        if self.new_features and self.new_labels:
            new_data = {"features": self.new_features, "labels": self.new_labels}
            with open("new_test_data.pkl", "wb") as f:
                pickle.dump(new_data, f)
            self.eval_result_label.setText("Nye testdata lagret til new_test_data.pkl!")
            self.save_data_button.setEnabled(False)
        else:
            self.eval_result_label.setText("Ingen nye data å lagre.")

    def closeEvent(self, event):
        event.accept()

# Hovedprogram
def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    try:
        print("Starter MindRove-board...")
        params = MindRoveInputParams()
        params.ip_address = "192.168.4.1"
        params.ip_port = 4210
        board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
        print("Forbereder sesjon...")
        board_shim.prepare_session()
        print("Starter stream...")
        board_shim.start_stream()

        app = QtWidgets.QApplication(sys.argv)
        classifier = NeuralNetworkClassifier()
        sensor_data = SensorData(board_shim)

        def handle_connection_lost(message):
            QtWidgets.QMessageBox.critical(None, "Tilkoblingsfeil", message + "\nSjekk nettverkstilkoblingen og start programmet på nytt.")
            app.quit()

        sensor_data.connection_lost.connect(handle_connection_lost)

        model_file = "mlp_model.pkl"
        use_saved = False

        if os.path.exists(model_file):
            dialog = QtWidgets.QMessageBox()
            dialog.setWindowTitle("Velg Modus")
            dialog.setText("En lagret modell ble funnet. Vil du bruke den for live testing eller starte en ny kalibrering?")
            dialog.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            result = dialog.exec_()
            if result == QtWidgets.QMessageBox.Yes:
                classifier.load_model(model_file)
                test_window = TestingWindow(classifier, sensor_data, CalibrationWindow(classifier, sensor_data))
                test_window.show()
                use_saved = True
            else:
                if os.path.exists(model_file):
                    os.remove(model_file)
                print("Eksisterende modell slettet, starter ny kalibrering.")

        if not use_saved:
            calibration_window = CalibrationWindow(classifier, sensor_data)
            calibration_window.show()
            print("Viser GUI-vindu for ny kalibrering...")

        sys.exit(app.exec_())
    except Exception as e:
        print(f"Kritisk feil i main: {str(e)}")
        import traceback
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(None, "Kritisk Feil", f"Kunne ikke starte programmet: {str(e)}\nSjekk nettverk og MindRove-enhet.")
        sys.exit(1)

if __name__ == '__main__':
    main()