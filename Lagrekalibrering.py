import logging
import numpy as np
import time
import json
import os
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
from PyQt5.QtWidgets import QMessageBox, QProgressBar, QInputDialog, QLabel, QPushButton, QFileDialog
from sklearn.neural_network import MLPClassifier
from collections import deque, Counter
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt, welch
from scipy.stats import pearsonr

# Filtreringsfunksjoner
def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

class Graph(QtWidgets.QMainWindow):
    def __init__(self, board_shim):
        super().__init__()
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.accel_channels = [8, 9, 10]
        self.gyro_channels = [11, 12, 13]
        self.mag_channels = [14, 15, 16]
        
        # logging.info(f"EMG-kanaler: {self.exg_channels}")
        # logging.info(f"Forventede akselerometerkanaler: {self.accel_channels}")
        # logging.info(f"Forventede gyroskopkanaler: {self.gyro_channels}")
        # logging.info(f"Forventede magnetometerkanaler: {self.mag_channels}")
        
        self.model = None
        self.training_features = []
        self.training_labels = []
        self.initial_features = {}
        self.dynamic_energy_threshold = 50
        self.window_duration = 1
        self.last_log_time = time.time()

        self.smoothing_window = 5
        self.min_energy_threshold = 40
        self.consecutive_frames_needed = 2

        self.last_prediction = 0
        self.prediction_count = 0
        self.default_class = 0
        self.current_probs = None
        self.is_closed = False  # Tilstandsvariabel for knytneve
        self.calibration_type = None
        self.calibration_file = "calibration_data.json"
        self.fist_detected = False  # Hjelpevariabel for å spore knytneve-deteksjon
        self.prediction_buffer = deque(maxlen=8)  # Buffer for glidende gjennomsnitt
        self.current_movement = self.default_class  # Start med "Hånd Stille"

        self.current_pwm = {'H': 1500, 'T': 1500, 'I': 1500, 'M': 1500, 'R': 1500, 'P': 1500}

        self.label_mapping = {
            0: ('H', 1500),   # Hånd Stille Flat (hvileposisjon på 1500)
            1: ('T', 2000),   # Tommel Flat (2000 pga motsatt retning)
            2: ('I', 1000),   # Pekefinger Flat (1000)
            3: ('M', 1000),   # Langfinger Flat (1000)
            4: ('R', 1000),   # Ringfinger Flat (1000)
            5: ('P', 1000),   # Lillefinger Flat (1000)
            6: ('T', 1000),   # Tommel Rotert (1000)
            7: ('I', 2000),   # Pekefinger Rotert (2000)
            8: ('M', 2000),   # Langfinger Rotert (2000)
            9: ('R', 2000),   # Ringfinger Rotert (2000)
            10: ('P', 2000),  # Lillefinger Rotert (2000)
            11: ('H', 1500),  # Hånd Stille Rotert (hvileposisjon)
            12: ('G', None),  # Knytneve (spesialhåndtering: veksler åpen/lukket)
            13: ('R', 1500),  # Rotasjon Flat til Rotert (hvileposisjon)
        }

        self.classes_to_calibrate = [
            ("Hånd Stille Flat", 0), ("Tommel Flat", 1), ("Pekefinger Flat", 2),
            ("Langfinger Flat", 3), ("Ringfinger Flat", 4), ("Lillefinger Flat", 5),
            ("Hånd Stille Rotert", 11), ("Tommel Rotert", 6), ("Pekefinger Rotert", 7),
            ("Langfinger Rotert", 8), ("Ringfinger Rotert", 9), ("Lillefinger Rotert", 10),
            ("Knytneve", 12), ("Rotasjon Flat til Rotert", 13),
        ]

        self.init_ui()
        self.load_calibration()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def init_ui(self):
        self.resize(1450, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.prediction_label = QLabel("Prediksjon: Venter på data...", self)
        self.layout.addWidget(self.prediction_label)
        self.prob_label = QLabel("Sannsynligheter: Ikke tilgjengelig", self)
        self.layout.addWidget(self.prob_label)
        self.hand_state_label = QLabel("Håndens tilstand: Ikke tilgjengelig", self)
        self.layout.addWidget(self.hand_state_label)

        self.timer_label = QtWidgets.QLabel("Tid igjen: -", self)
        self.layout.addWidget(self.timer_label)
        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        self.win1 = pg.GraphicsLayoutWidget(title="Mindrove Data: Channels 1-4")
        self.layout.addWidget(self.win1)
        self.win2 = pg.GraphicsLayoutWidget(title="Mindrove Data: Channels 5-8")
        self.layout.addWidget(self.win2)

        self.plots_1_4, self.curves_1_4 = [], []
        self.plots_5_8, self.curves_5_8 = [], []
        self.init_plots()

        self.calibrate_button = QPushButton("Start Kalibrering")
        self.calibrate_button.clicked.connect(self.calibrate_all_fingers)
        self.layout.addWidget(self.calibrate_button)

        self.adjust_button = QPushButton("Juster Kalibrering")
        self.adjust_button.clicked.connect(self.adjust_calibration)
        self.layout.addWidget(self.adjust_button)

        self.save_button = QPushButton("Lagre Kalibrering")
        self.save_button.clicked.connect(self.save_calibration)
        self.layout.addWidget(self.save_button)

        self.load_json_button = QPushButton("Last JSON-kalibrering")
        self.load_json_button.clicked.connect(self.load_json_calibration)
        self.layout.addWidget(self.load_json_button)

        self.corr_button = QPushButton("Vis Korrelasjonsmatrise")
        self.corr_button.clicked.connect(self.plot_feature_correlation)
        self.layout.addWidget(self.corr_button)

        self.show()

    def set_pwm(self, finger, target_pwm, prediction_confidence):
        if finger == 'G' and self.calibration_type == "Knytneve" and prediction_confidence > 0.8:
            if not self.fist_detected:  # Ny deteksjon
                self.is_closed = not self.is_closed
                self.fist_detected = True
                if self.is_closed:
                    # logging.info("Lukker hånden")
                    self.hand_state_label.setText("Håndens tilstand: Lukket")
                    finger_targets = {
                        'T': 1000,  # Tommel lukket
                        'I': 2000,  # Pekefinger lukket
                        'M': 2000,  # Langfinger lukket
                        'R': 2000,  # Ringfinger lukket
                        'P': 2000,  # Lillefinger lukket
                    }
                else:
                    # logging.info("Åpner hånden")
                    self.hand_state_label.setText("Håndens tilstand: Åpen")
                    finger_targets = {
                        'T': 1500,  # Nøytral posisjon
                        'I': 1500,
                        'M': 1500,
                        'R': 1500,
                        'P': 1500,
                    }
                for f, target in finger_targets.items():
                    if self.current_pwm[f] != target:
                        self.current_pwm[f] = target
                        # logging.info(f"Simulert PWM for {f} satt til {target}")
        elif finger in self.current_pwm and self.calibration_type != "Knytneve":
            self.current_pwm[finger] = target_pwm
            # logging.info(f"Simulert PWM for {finger} satt til {target_pwm}")
            for other_finger in self.current_pwm:
                if other_finger != finger and self.current_pwm[other_finger] != 1500:
                    self.current_pwm[other_finger] = 1500
                    # logging.info(f"Simulert PWM for {other_finger} satt til 1500 (hvile)")
            self.hand_state_label.setText("Håndens tilstand: Ikke tilgjengelig")
        elif finger == 'H' or finger == 'R':
            if self.calibration_type != "Knytneve" or not self.is_closed:
                for f in self.current_pwm:
                    if self.current_pwm[f] != 1500:
                        self.current_pwm[f] = 1500
                        # logging.info(f"Simulert PWM for {f} satt til 1500 (hvile)")
                self.hand_state_label.setText("Håndens tilstand: Ikke tilgjengelig")
            elif self.is_closed:
                self.hand_state_label.setText("Håndens tilstand: Lukket")
        # else:
            # logging.warning(f"Ugyldig finger: {finger}")

    def load_json_calibration(self):
        reply = QMessageBox.question(self, "Last JSON", "Vil du laste inn en eksisterende JSON-kalibreringsfil?", 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            fname, _ = QFileDialog.getOpenFileName(self, "Velg JSON-fil", "", "JSON Files (*.json)")
            if fname:
                try:
                    with open(fname, 'r') as f:
                        data = json.load(f)
                    self.initial_features = {int(k): [np.array(feat) for feat in v] for k, v in data["initial_features"].items()}
                    self.training_features = [np.array(feat) for feat in data["training_features"]]
                    self.training_labels = data["training_labels"]
                    self.dynamic_energy_threshold = data["dynamic_energy_threshold"]
                    self.calibration_type = data["calibration_type"]
                    self.default_class = data["default_class"]
                    self.train_model(self.training_features, self.training_labels)
                    # logging.info(f"Kalibreringsdata lastet fra {fname} og modell trent.")
                    QMessageBox.information(self, "Lastet", f"Kalibreringsdata lastet fra {fname}.")
                    if self.calibration_type == "Knytneve":
                        self.is_closed = False
                        self.hand_state_label.setText("Håndens tilstand: Åpen")
                        for f in self.current_pwm:
                            self.current_pwm[f] = 1500
                    else:
                        self.hand_state_label.setText("Håndens tilstand: Ikke tilgjengelig")
                except Exception as e:
                    # logging.error("Feil ved lasting av JSON-fil:", exc_info=True)
                    QMessageBox.critical(self, "Feil", "Kunne ikke laste JSON-fil.")

    def save_calibration(self):
        if not self.training_features or not self.training_labels:
            QMessageBox.warning(self, "Ingen data", "Ingen kalibreringsdata å lagre.")
            return
        data_to_save = {
            "initial_features": {str(k): v for k, v in self.initial_features.items()},
            "training_features": self.training_features,
            "training_labels": self.training_labels,
            "dynamic_energy_threshold": self.dynamic_energy_threshold,
            "calibration_type": self.calibration_type,
            "default_class": self.default_class
        }
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(data_to_save, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            QMessageBox.information(self, "Lagret", "Kalibreringsdata er lagret til calibration_data.json.")
            # logging.info("Kalibreringsdata lagret.")
        except Exception as e:
            # logging.error("Feil ved lagring av kalibreringsdata:", exc_info=True)
            QMessageBox.critical(self, "Feil", "Kunne ikke lagre kalibreringsdata.")

    def load_calibration(self):
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                self.initial_features = {int(k): [np.array(feat) for feat in v] for k, v in data["initial_features"].items()}
                self.training_features = [np.array(feat) for feat in data["training_features"]]
                self.training_labels = data["training_labels"]
                self.dynamic_energy_threshold = data["dynamic_energy_threshold"]
                self.calibration_type = data["calibration_type"]
                self.default_class = data["default_class"]
                self.train_model(self.training_features, self.training_labels)
                # logging.info("Kalibreringsdata lastet og modell trent.")
                QMessageBox.information(self, "Lastet", "Kalibreringsdata er lastet fra fil.")
                if self.calibration_type == "Knytneve":
                    self.is_closed = False
                    self.hand_state_label.setText("Håndens tilstand: Åpen")
                    for f in self.current_pwm:
                        self.current_pwm[f] = 1500
                else:
                    self.hand_state_label.setText("Håndens tilstand: Ikke tilgjengelig")
            except Exception as e:
                # logging.error("Feil ved lasting av kalibreringsdata:", exc_info=True)
                QMessageBox.critical(self, "Feil", "Kunne ikke laste kalibreringsdata.")

    def prompt_user(self, message):
        msg = QMessageBox()
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return msg.exec_()

    def start_measurement(self, class_name, duration):
        # logging.info(f"Starter måling for {class_name}...")
        self.progress_bar.setMaximum(duration)
        self.progress_bar.setValue(0)
        data = []
        for t in range(duration, 0, -1):
            self.timer_label.setText(f"{class_name}: {t} sekunder igjen...")
            self.progress_bar.setValue(duration - t + 1)
            QtWidgets.QApplication.processEvents()
            num_samples = int(self.sampling_rate * self.window_duration)
            new_data = self.board_shim.get_current_board_data(num_samples)
            if new_data.size > 0:
                data.append(new_data)
            QtCore.QThread.msleep(1000)
        self.timer_label.setText(f"{class_name}: Ferdig!")
        self.progress_bar.setValue(0)
        QtWidgets.QApplication.processEvents()
        return np.concatenate(data, axis=1) if data else np.array([])

    def calibrate_finger(self, class_name, label):
        duration = 45 if label in [0, 11] else 45 # Økt varighet for Hånd Stille
        while True:
            if self.prompt_user(f"Trykk OK for å starte måling av {class_name}.") == QMessageBox.Ok:
                data = self.start_measurement(class_name, duration)
                if data.size > 0:
                    if self.prompt_user("Er du fornøyd med resultatet?") == QMessageBox.Ok:
                        return data

    def calibrate_all_fingers(self):
        calibration_type, _ = QInputDialog.getItem(self, "Velg kalibreringstype", "Velg type:",
                                                   ["Flat Hånd", "Rotert Hånd", "Knytneve", "Full Kalibrering"], 0, False)
        self.calibration_type = calibration_type

        if calibration_type == "Flat Hånd":
            classes = self.classes_to_calibrate[:6]
            self.default_class = 0
        elif calibration_type == "Rotert Hånd":
            classes = self.classes_to_calibrate[6:12]
            self.default_class = 11
        elif calibration_type == "Knytneve":
            classes = [self.classes_to_calibrate[0], self.classes_to_calibrate[6], self.classes_to_calibrate[12], self.classes_to_calibrate[13]]
            self.default_class = 0
            self.is_closed = False
            self.hand_state_label.setText("Håndens tilstand: Åpen")
            for f in self.current_pwm:
                self.current_pwm[f] = 1500
        else:
            classes = [cls for cls in self.classes_to_calibrate if cls[1] != 12]
            self.default_class = 0

        self.initial_features = {}
        X = []
        y = []
        energies = []
        for class_name, label in classes:
            data = self.calibrate_finger(class_name, label)
            if data.size > 0:
                processed_data = self.preprocess_data(data)
                window_size = int(self.sampling_rate * self.window_duration)
                segments = self.segment_data(processed_data, window_size)
                class_features = []
                for segment in segments:
                    features = self.extract_features_from_window(segment)
                    X.append(features)
                    y.append(label)
                    class_features.append(features)
                    energy = np.mean([np.mean(ch**2) for ch in segment[:8]])
                    energies.append(energy)
                self.initial_features[label] = class_features
        
        self.dynamic_energy_threshold = np.mean(energies) * 0.8
        # logging.info(f"Dynamisk energi-terskel satt til: {self.dynamic_energy_threshold}")
        
        X_balanced, y_balanced = self.balance_training_data(X, y, max_samples_per_class=100)
        self.training_features = X_balanced
        self.training_labels = y_balanced
        # logging.info(f"Klassefordeling i treningsdata: {Counter(y_balanced)}")
        self.train_model(X_balanced, y_balanced)
        # logging.info("Modell ferdig trent.")

    def calculate_similarity(self, initial_features, adjustment_features):
        if len(initial_features) == 0 or len(adjustment_features) == 0:
            return 0.0
        initial_array = np.array(initial_features)
        adjust_array = np.array(adjustment_features)
        initial_mean = np.mean(initial_array, axis=0)
        adjust_mean = np.mean(adjust_array, axis=0)
        correlation, _ = pearsonr(initial_mean, adjust_mean)
        similarity = max(0, correlation) * 100
        return similarity

    def balance_training_data(self, X, y, max_samples_per_class=100):
        X_balanced = []
        y_balanced = []
        # logging.info(f"Før balansering - klassefordeling: {Counter(y)}")
        for label in set(y):
            indices = [i for i, l in enumerate(y) if l == label]
            num_samples = len(indices)
            if num_samples > max_samples_per_class:
                indices = np.random.choice(indices, max_samples_per_class, replace=False)
            elif num_samples < max_samples_per_class and label in [0, 11]:  # Oversample Hånd Stille
                indices = np.random.choice(indices, max_samples_per_class, replace=True)
            for idx in indices:
                X_balanced.append(X[idx])
                y_balanced.append(y[idx])
        # logging.info(f"Etter balansering - klassefordeling: {Counter(y_balanced)}")
        return X_balanced, y_balanced

    def adjust_calibration(self):
        if self.calibration_type is None:
            QMessageBox.warning(self, "Ingen kalibrering", "Du må gjennomføre en kalibrering eller laste en JSON-fil først.")
            return
        
        if self.calibration_type == "Knytneve":
            QMessageBox.information(self, "Ingen justering", "Knytneve-kalibrering støtter ikke justering.")
            return

        flat_hand_adjustments = [
            ("Hånd Stille Flat", 0), ("Tommel Flat", 1), ("Pekefinger Flat", 2),
            ("Langfinger Flat", 3), ("Ringfinger Flat", 4), ("Lillefinger Flat", 5),
        ]
        rotated_hand_adjustments = [
            ("Hånd Stille Rotert", 11), ("Tommel Rotert", 6), ("Pekefinger Rotert", 7),
            ("Langfinger Rotert", 8), ("Ringfinger Rotert", 9), ("Lillefinger Rotert", 10),
        ]

        if self.calibration_type == "Flat Hånd":
            adjustments = flat_hand_adjustments
        elif self.calibration_type == "Rotert Hånd":
            adjustments = rotated_hand_adjustments
        elif self.calibration_type == "Full Kalibrering":
            adjustments = flat_hand_adjustments + rotated_hand_adjustments
        else:
            return

        items = [item[0] for item in adjustments]
        movement, ok = QInputDialog.getItem(self, "Juster kalibrering", "Velg bevegelse å justere:", items, 0, False)
        if ok and movement:
            label = [item[1] for item in adjustments if item[0] == movement][0]
            new_data = self.calibrate_finger(movement, label)
            if new_data.size > 0:
                processed_data = self.preprocess_data(new_data)
                window_size = int(self.sampling_rate * self.window_duration)
                segments = self.segment_data(processed_data, window_size)
                adjustment_features = []
                for segment in segments:
                    features = self.extract_features_from_window(segment)
                    adjustment_features.append(features)
                
                self.initial_features[label] = adjustment_features
                X = []
                y = []
                for lbl, feats in self.initial_features.items():
                    X.extend(feats)
                    y.extend([lbl] * len(feats))
                
                similarity = self.calculate_similarity(self.initial_features.get(label, []), adjustment_features)
                # logging.info(f"Likhet for {movement}: {similarity:.2f}%")
                self.prob_label.setText(f"Likhet: {similarity:.2f}%")
                
                X_balanced, y_balanced = self.balance_training_data(X, y)
                self.training_features = X_balanced
                self.training_labels = y_balanced
                self.train_model(X_balanced, y_balanced)
                QMessageBox.information(self, "Kalibrering oppdatert", 
                                        f"Kalibreringen for {movement} er oppdatert (Likhet: {similarity:.2f}%).")

    def segment_data(self, data, window_size):
        n_channels, total_samples = data.shape
        segments = []
        for i in range(0, total_samples, window_size):
            end_idx = min(i + window_size, total_samples)
            segments.append(data[:, i:end_idx])
        return segments

    def extract_features_from_window(self, window_data):
        emg_data = window_data[:8]
        features = []
        for i, channel in enumerate(emg_data):
            if channel.size > 0:
                rms = np.sqrt(np.mean(channel**2))
                var = np.var(channel)
                wl = np.sum(np.abs(np.diff(channel))) / len(channel)
                zcr = np.sum(np.diff(np.sign(channel)) != 0) / len(channel)
                fft_vals = np.abs(np.fft.rfft(channel))
                freqs = np.fft.rfftfreq(len(channel), 1/self.sampling_rate)
                mean_freq = np.sum(fft_vals * freqs) / np.sum(fft_vals) if np.sum(fft_vals) != 0 else 0
                median_freq = freqs[np.where(np.cumsum(fft_vals) >= np.sum(fft_vals)/2)[0][0]] if np.sum(fft_vals) != 0 else 0
                cA, cD = pywt.dwt(channel, 'db4')
                extra = rms / (var + 1e-6)
                f, Pxx = welch(channel, fs=self.sampling_rate, nperseg=min(256, len(channel)), noverlap=min(128, len(channel)//2))
                spec_energy_20_70 = np.sum(Pxx[(f >= 20) & (f <= 70)]) if any((f >= 20) & (f <= 70)) else 0
                spec_energy_70_150 = np.sum(Pxx[(f > 70) & (f <= 150)]) if any((f > 70) & (f <= 150)) else 0
                freq_ratio = spec_energy_20_70 / spec_energy_70_150 if spec_energy_70_150 > 0 else 0
                diff1 = np.diff(channel)
                diff2 = np.diff(diff1)
                var_signal = var
                var_diff1 = np.var(diff1)
                var_diff2 = np.var(diff2)
                mobility = np.sqrt(var_diff1 / var_signal) if var_signal > 0 else 0
                complexity = (np.sqrt(var_diff2 / var_diff1) / mobility) if var_diff1 > 0 and mobility > 0 else 0
                features.extend([rms, var, wl, zcr, mean_freq, median_freq, np.mean(cA), np.std(cD), extra,
                                 freq_ratio, mobility, complexity])
            else:
                features.extend([0] * 12)
        return features

    def preprocess_data(self, data):
        processed_data = np.copy(data)
        if np.isnan(processed_data).any():
            processed_data = np.nan_to_num(processed_data)
        fs = self.sampling_rate
        
        if processed_data.shape[0] < max(self.mag_channels) + 1:
            # logging.warning(f"Data har færre kanaler ({processed_data.shape[0]}) enn forventet IMU-kanaler.")
            accel_data = np.zeros((3, processed_data.shape[1]))
            gyro_data = np.zeros((3, processed_data.shape[1]))
        else:
            accel_data = processed_data[self.accel_channels, :]
            gyro_data = processed_data[self.gyro_channels, :]
        
        for channel_idx in self.exg_channels:
            if processed_data[channel_idx].size > 0:
                DataFilter.detrend(processed_data[channel_idx], DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(processed_data[channel_idx], fs, 10.0, 450.0, 4,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.perform_bandstop(processed_data[channel_idx], fs, 48.0, 52.0, 3,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                processed_data[channel_idx] = moving_average(processed_data[channel_idx], window_size=5)
        
        return processed_data

    def train_model(self, X, y):
        if len(X) == 0 or len(set(y)) < 2:
            # logging.error("For få klasser til å trene modellen.")
            return
        try:
            X_array = np.array(X)
            # logging.info(f"Antall trekk i treningsdata (X): {X_array.shape[1]}")
            # logging.info(f"Antall eksempler i treningsdata (X): {X_array.shape[0]}")
            pipe = Pipeline([
                ('scaler', RobustScaler()),
                ('pca', PCA(n_components=20)),  # Redusert for å unngå overfitting
                ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.05, max_iter=2000, random_state=42))
            ])
            pipe.fit(X_array, y)
            self.model = pipe
            # logging.info("MLP-modell trent.")
        except Exception as e:
            # logging.error("Feil under modelltrening:", exc_info=True)
            pass

    def init_plots(self):
        self.plot_colors = ['y', 'g', 'b', 'r', 'c', 'm', 'w', 'y']
        fixed_plot_width = 350
        for i, channel in enumerate(self.exg_channels[:8]):
            if i < 4:
                plot = self.win1.addPlot(title=f"Kanal {channel+1}")
            else:
                plot = self.win2.addPlot(title=f"Kanal {channel+1}")
            plot.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
            plot.setMinimumWidth(fixed_plot_width)
            plot.setMaximumWidth(fixed_plot_width)
            x_range = int(self.sampling_rate * self.window_duration)
            plot.setXRange(0, x_range, padding=0)
            plot.getViewBox().enableAutoRange(axis='x', enable=False)
            plot.setYRange(-100, 100)
            curve = plot.plot(pen=self.plot_colors[i])
            if i < 4:
                self.plots_1_4.append(plot)
                self.curves_1_4.append(curve)
            else:
                self.plots_5_8.append(plot)
                self.curves_5_8.append(curve)

    def update(self):
        try:
            num_samples = int(self.sampling_rate * self.window_duration)
            data = self.board_shim.get_current_board_data(num_samples)
            if data.size > 0:
                processed_data = self.preprocess_data(data)
                for i, channel_data in enumerate(processed_data[:8]):
                    mean_value = np.mean(channel_data)
                    y_min = mean_value - 500
                    y_max = mean_value + 500
                    if i < 4:
                        self.plots_1_4[i].setYRange(y_min, y_max)
                        self.curves_1_4[i].setData(channel_data.tolist())
                    else:
                        idx = i - 4
                        self.plots_5_8[idx].setYRange(y_min, y_max)
                        self.curves_5_8[idx].setData(channel_data.tolist())

                if self.model is not None:
                    features = self.extract_features_from_window(processed_data)
                    features_array = np.array(features)
                    mean_energy = np.mean([np.mean(channel**2) for channel in processed_data[:8]])
                    # logging.info(f"Mean energy: {mean_energy}")
                    
                    # Dynamisk justering av energi-terskel
                    if mean_energy < self.min_energy_threshold:
                        temp_threshold = self.dynamic_energy_threshold * 0.5
                    else:
                        temp_threshold = self.dynamic_energy_threshold
                    
                    probs = self.model.predict_proba([features_array])[0]
                    # logging.info(f"Rå prediksjonssannsynligheter: {probs}")
                    final_prediction = self.model.classes_[np.argmax(probs)]
                    prediction_confidence = max(probs)
                    # logging.info(f"Final prediction: {final_prediction} (Confidence: {prediction_confidence:.2f})")

                    # Legg til prediksjon i buffer kun hvis energi er over terskel
                    if mean_energy > temp_threshold:
                        self.prediction_buffer.append((final_prediction, prediction_confidence))
                    else:
                        self.prediction_buffer.append((self.default_class, 1.0))

                    pred_label, pwm_value = self.label_mapping.get(final_prediction, ('H', 1500))
                    class_name = [name for name, label in self.classes_to_calibrate if label == final_prediction][0]
                    self.prediction_label.setText(f"Prediksjon: {class_name} ({pred_label}, Klasse {final_prediction})")
                    self.prob_label.setText(f"Sannsynligheter: {', '.join([f'{p:.2f}' for p in probs])}")
                    self.current_probs = probs

                    if len(self.prediction_buffer) == 8:
                        # Velg prediksjon basert på glidende gjennomsnitt, kun konfidens > 0.7
                        valid_predictions = [(pred, conf) for pred, conf in self.prediction_buffer if conf > 0.7]
                        if valid_predictions:
                            most_common = Counter([pred for pred, _ in valid_predictions]).most_common(1)[0]
                            selected_prediction = most_common[0]
                            confidence = most_common[1] / len(valid_predictions)
                            if confidence < 0.7:  # Streng terskel
                                selected_prediction = self.default_class
                        else:
                            selected_prediction = self.default_class

                        if selected_prediction != self.current_movement:
                            self.current_movement = selected_prediction
                            pred_label, pwm_value = self.label_mapping.get(selected_prediction, ('H', 1500))
                            self.set_pwm(pred_label, pwm_value, prediction_confidence)
                            if selected_prediction != 12:
                                self.fist_detected = False

            QtWidgets.QApplication.processEvents()

        except Exception as e:
            # logging.error("Feil i update-metoden:", exc_info=True)
            pass

    def plot_feature_correlation(self):
        if not self.training_features or len(self.training_features) == 0:
            QMessageBox.warning(self, "Ingen data", "Ingen treningsdata tilgjengelig for korrelasjonsmatrise.")
            return
        features_array = np.array(self.training_features)
        features_per_channel = 12
        num_channels = 8
        base_labels = ["RMS", "VAR", "WL", "ZCR", "MEAN_FREQ", "MEDIAN_FREQ", "MEAN_C_A", "STD_C_D", 
                       "RMS/VAR", "FREQ_RATIO", "MOBILITY", "COMPLEXITY"]
        feature_names = [f"Ch{ch+1}_{label}" for ch in range(num_channels) for label in base_labels]
        df_features = pd.DataFrame(features_array, columns=feature_names)
        corr_matrix = df_features.corr()
        plt.figure(figsize=(25, 20))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 6})
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def closeEvent(self, event):
        self.board_shim.release_session()
        for f in self.current_pwm:
            self.current_pwm[f] = 1500
            # logging.info(f"Simulert PWM for {f} tilbakestilt til 1500")
        # logging.info("Økt avsluttet.")
        event.accept()

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s', handlers=[
        logging.FileHandler("calibration.log")
        # logging.StreamHandler()
    ])
    BoardShim.enable_dev_board_logger()
    params = MindRoveInputParams()
    try:
        board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
        board_shim.prepare_session()
        board_shim.start_stream()
        app = QtWidgets.QApplication([])
        window = Graph(board_shim)
        app.exec_()
    except Exception as e:
        # logging.error("Feil i hovedprogrammet:", exc_info=True)
        pass

if __name__ == "__main__":
    main()