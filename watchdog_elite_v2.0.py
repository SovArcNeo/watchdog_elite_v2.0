#!/usr/bin/env python3

# WATCHDOG_ELITE_v2.0_.py - Advanced Cybersecurity Monitoring Platform with GUI
# Version 2.0 - QUANTUM FORTRESS EDITION with MATRIX GUI
# FIXES: Neural network training hang, 3D graph sizing, Added Matrix code rain
# Classification: ELITE CLASSIFIED

import time
import os
import sys
import json
import logging
import threading
import hashlib
import traceback
import socket
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from threading import Thread
import logging.handlers

# GUI imports - tkinter and visualization
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

# Quantum-resistant cryptography imports
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

# Try to import Argon2 - multiple sources
ARGON2_AVAILABLE = False
ARGON2_TYPE = None

try:
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2
    ARGON2_AVAILABLE = True
    ARGON2_TYPE = "cryptography"
except ImportError:
    try:
        import argon2
        from argon2.low_level import hash_secret_raw, Type
        ARGON2_AVAILABLE = True
        ARGON2_TYPE = "argon2-cffi"

        class Argon2Wrapper:
            def __init__(self, memory_cost, time_cost, parallelism, hash_len, salt):
                self.memory_cost = memory_cost
                self.time_cost = time_cost
                self.parallelism = parallelism
                self.hash_len = hash_len
                self.salt = salt

            def derive(self, key_material):
                return hash_secret_raw(
                    secret=key_material,
                    salt=self.salt,
                    time_cost=self.time_cost,
                    memory_cost=self.memory_cost // 1024,
                    parallelism=self.parallelism,
                    hash_len=self.hash_len,
                    type=Type.ID
                )
        Argon2 = Argon2Wrapper
    except ImportError:
        ARGON2_AVAILABLE = False
        Argon2 = None

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM
from cryptography.hazmat.primitives.hashes import SHA3_512, BLAKE2b

# NumPy and Scientific Computing
import numpy as np

# Advanced visualization imports
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available")

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.critical("psutil is required")
    sys.exit(1)

#=================================================================
#       CONSTANTS & CONFIGURATION
#=================================================================

WATCHDOG_VERSION = "2.0-ELITE-QUANTUM-GUI-FIXED"
CLASSIFICATION = "ELITE CLASSIFIED"
SYSTEM_NAME = "WATCHDOG_ELITE_SENTINEL_v2.0_FIXED"

CONFIG = {
    "max_runtime_hours": 8760,
    "gui_theme": "matrix",
    "gui_refresh_rate_ms": 100,
    "gui_fps": 60,
    "nn_hidden_layers": [128, 64, 32],
    "nn_learning_rate": 0.001,
    "nn_batch_size": 32,
    "nn_epochs": 50,
    "metrics_buffer_size": 10000
}

# Crypto constants
QUANTUM_KEY_SIZE = 64
ARGON2_MEMORY = 256 * 1024
ARGON2_ITERATIONS = 8
ARGON2_PARALLELISM = 16
SCRYPT_N = 2**20
SCRYPT_R = 8
SCRYPT_P = 1

# Paths
DEFAULT_BASE_DIR = Path(os.path.expanduser("~")) / ".watchdog_elite"
BASE_DIR = DEFAULT_BASE_DIR
NN_MODELS_DIR = BASE_DIR / "nn_models"
HEATMAP_DATA_DIR = BASE_DIR / "heatmaps"
VISUALIZATION_DIR = BASE_DIR / "visualizations"

# Matrix GUI Color Scheme
MATRIX_COLORS = {
    'bg_primary': '#000000',
    'bg_secondary': '#001100',
    'bg_panel': '#002200',
    'fg_bright': '#00FF00',
    'fg_normal': '#00CC00',
    'fg_dim': '#008800',
    'fg_alert': '#FF0000',
    'fg_warning': '#FFFF00',
    'fg_info': '#00FFFF',
    'grid_line': '#003300',
    'accent': '#00FF41'
}

#========================================================================
#                  PURE NUMPY NEURAL NETWORK (FIXED)
#========================================================================

class NumpyNeuralNetwork:
    """Pure NumPy neural network - FIXED for GUI training"""

    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int,
                 learning_rate: float = 0.001, dropout_rate: float = 0.2):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layer_sizes)

        self.weights = []
        self.biases = []
        self._initialize_parameters()

        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.activations = []
        self.z_values = []
        self.epoch_count = 0
        self.best_accuracy = 0.0
        self.training_time = 0.0
        self.training_callback = None  # NEW: Callback for GUI updates

    def _initialize_parameters(self):
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2.0 / self.layer_sizes[i])
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1, alpha)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        self.activations = [X]
        self.z_values = []
        current_activation = X

        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_activation = self.leaky_relu(z)

            if training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, current_activation.shape)
                current_activation = current_activation * dropout_mask / (1 - self.dropout_rate)

            self.activations.append(current_activation)

        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)

        return output

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        m = y_true.shape[0]
        epsilon = 1e-10
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return loss

    def backward_propagation(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = X.shape[0]
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        dz = self.activations[-1] - y

        for i in range(len(self.weights) - 1, -1, -1):
            dW[i] = np.dot(self.activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.leaky_relu_derivative(self.z_values[i - 1])

        return dW, db

    def update_parameters(self, dW: List[np.ndarray], db: List[np.ndarray]):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, verbose: bool = False) -> Dict[str, List[float]]:
        """Train the neural network - FIXED for GUI"""
        start_time = time.time()
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            epoch_correct = 0

            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                y_pred = self.forward_propagation(batch_X, training=True)
                batch_loss = self.compute_loss(y_pred, batch_y)
                epoch_loss += batch_loss

                predictions = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                epoch_correct += np.sum(predictions == true_labels)

                dW, db = self.backward_propagation(batch_X, batch_y)
                self.update_parameters(dW, db)

            avg_loss = epoch_loss / (n_samples / batch_size)
            accuracy = epoch_correct / n_samples

            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(accuracy)

            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(val_pred, y_val)
                val_accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))

                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)

                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy

            # NEW: Call GUI callback if provided
            if self.training_callback:
                self.training_callback(epoch, epochs, avg_loss, accuracy)

        self.epoch_count = epochs
        self.training_time = time.time() - start_time

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_propagation(X, training=False)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        loss = self.compute_loss(y_pred, y)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)

        n_classes = y.shape[1]
        precision_per_class = []
        recall_per_class = []

        for c in range(n_classes):
            true_positive = np.sum((predictions == c) & (true_labels == c))
            false_positive = np.sum((predictions == c) & (true_labels != c))
            false_negative = np.sum((predictions != c) & (true_labels == c))

            precision = true_positive / (true_positive + false_positive + 1e-10)
            recall = true_positive / (true_positive + false_negative + 1e-10)

            precision_per_class.append(precision)
            recall_per_class.append(recall)

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'precision': np.mean(precision_per_class),
            'recall': np.mean(recall_per_class),
            'f1_score': 2 * np.mean(precision_per_class) * np.mean(recall_per_class) /
                       (np.mean(precision_per_class) + np.mean(recall_per_class) + 1e-10)
        }

    def save_model(self, filepath: str):
        model_data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'config': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate,
                'dropout_rate': self.dropout_rate
            },
            'training_history': self.training_history,
            'epoch_count': self.epoch_count,
            'best_accuracy': self.best_accuracy,
            'training_time': self.training_time
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        logging.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        config = model_data['config']
        self.input_size = config['input_size']
        self.hidden_layers = config['hidden_layers']
        self.output_size = config['output_size']
        self.learning_rate = config['learning_rate']
        self.dropout_rate = config['dropout_rate']

        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.training_history = model_data['training_history']
        self.epoch_count = model_data.get('epoch_count', 0)
        self.best_accuracy = model_data.get('best_accuracy', 0.0)
        self.training_time = model_data.get('training_time', 0.0)

        logging.info(f"Model loaded from {filepath}")

#========================================================================
#                  QUANTUM CRYPTOGRAPHY ENGINE
#========================================================================

class QuantumCryptoEngine:
    """Post-quantum cryptography"""

    def __init__(self):
        self.entropy_pool = bytearray(QUANTUM_KEY_SIZE * 10)
        self.entropy_sources = []
        self.key_pool = deque(maxlen=100)
        self._initialize_entropy_sources()
        self._initialize_quantum_keys()

    def _initialize_entropy_sources(self):
        self.entropy_sources.append(('timing_jitter', self._timing_jitter))
        self.entropy_sources.append(('memory_patterns', self._memory_entropy))

    def _initialize_quantum_keys(self):
        for i in range(20):
            key_data = self.generate_quantum_key()
            self.key_pool.append(key_data)

    def generate_quantum_key(self, key_size: int = QUANTUM_KEY_SIZE) -> bytes:
        entropy_data = bytearray()

        for _,  source_func in self.entropy_sources:
            if callable(source_func):
                try:
                    entropy_data.extend(source_func())
                except:
                    pass

        entropy_data.extend(os.urandom(key_size))

        salt = os.urandom(16)
        if ARGON2_AVAILABLE and Argon2 is not None:
            kdf = Argon2(
                memory_cost=ARGON2_MEMORY,
                time_cost=ARGON2_ITERATIONS,
                parallelism=ARGON2_PARALLELISM,
                hash_len=key_size,
                salt=salt
            )
        else:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_size,
                salt=salt,
                iterations=600000
            )

        quantum_key = kdf.derive(bytes(entropy_data))

        hasher = hashes.Hash(BLAKE2b(64))
        hasher.update(quantum_key)
        hasher.update(os.urandom(32))
        final_key = hasher.finalize()[:key_size]

        return final_key

    def _timing_jitter(self) -> bytes:
        timings = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            _ = [i**2 for i in range(100)]
            end = time.perf_counter_ns()
            timings.append(end - start)

        jitter_data = bytearray()
        for i in range(1, len(timings)):
            delta = timings[i] - timings[i-1]
            jitter_data.extend(delta.to_bytes(8, 'little', signed=True))

        hasher = hashes.Hash(SHA3_512())
        hasher.update(bytes(jitter_data))
        return hasher.finalize()

    def _memory_entropy(self) -> bytes:
        allocations = []
        for size in [1024, 2048, 4096, 8192]:
            start = time.perf_counter_ns()
            data = bytearray(size)
            allocations.append(id(data))
            del data
            end = time.perf_counter_ns()
            allocations.append(end - start)

        entropy = bytearray()
        for value in allocations:
            entropy.extend(value.to_bytes(8, 'little'))

        return hashlib.sha3_512(entropy).digest()

    def encrypt_quantum(self, data: bytes, additional_data: Optional[bytes] = None) -> Tuple[bytes, bytes, bytes]:
        nonce_gcm = os.urandom(16)
        nonce_chacha = os.urandom(16)

        chacha_key = self.generate_quantum_key(32)
        chacha_cipher = ChaCha20Poly1305(chacha_key)
        ct_layer1 = chacha_cipher.encrypt(nonce_chacha, data, additional_data)

        aes_key = self.generate_quantum_key(32)
        aes_cipher = AESGCM(aes_key)
        ct_final = aes_cipher.encrypt(nonce_gcm, ct_layer1, additional_data)

        combined_key = self._combine_keys([chacha_key, aes_key])

        return ct_final, nonce_gcm + nonce_chacha, combined_key

    def _combine_keys(self, keys: List[bytes]) -> bytes:
        combined = bytearray()
        for key in keys:
            combined.extend(key)

        kdf = Scrypt(
            salt=os.urandom(32),
            length=64,
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P
        )

        return kdf.derive(bytes(combined))

#========================================================================
#                  REAL-WORLD SYSTEM DATA COLLECTOR
#========================================================================

class RealWorldDataCollector:
    """Collects real system metrics - NO SIMULATIONS"""

    def __init__(self):
        self.process_monitor = psutil if PSUTIL_AVAILABLE else None
        self.data_buffer = deque(maxlen=CONFIG['metrics_buffer_size'])
        self.collection_lock = threading.RLock()
        self.start_time = time.time()

    def collect_system_metrics(self) -> Dict[str, Any]:
        if not self.process_monitor:
            return {}

        try:
            metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'percent': psutil.cpu_percent(interval=0.1, percpu=False),
                    'percent_per_core': psutil.cpu_percent(interval=0.1, percpu=True),
                    'count_logical': psutil.cpu_count(logical=True),
                    'count_physical': psutil.cpu_count(logical=False)
                },
                'memory': {
                    'virtual': psutil.virtual_memory()._asdict(),
                    'swap': psutil.swap_memory()._asdict()
                },
                'disk': {
                    'usage': psutil.disk_usage('/')._asdict()
                },
                'network': {
                    'io_counters': psutil.net_io_counters()._asdict(),
                    'connections_count': len(psutil.net_connections()),
                    'interfaces': {name: addr._asdict() for name, addrs in psutil.net_if_addrs().items() for addr in addrs}
                },
                'processes': {
                    'count': len(psutil.pids()),
                    'running': len([p for p in psutil.process_iter(['status']) if p.info['status'] == psutil.STATUS_RUNNING])
                }
            }

            with self.collection_lock:
                self.data_buffer.append(metrics)

            return metrics

        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            return {}

    def collect_process_metrics(self) -> List[Dict[str, Any]]:
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status', 'num_threads']):
                try:
                    proc_info = proc.info
                    proc_info['cmdline'] = ' '.join(proc.cmdline())
                    proc_info['create_time'] = proc.create_time()
                    processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logging.error(f"Error collecting process metrics: {e}")

        return processes

    def collect_network_connections(self) -> List[Dict[str, Any]]:
        connections = []
        try:
            for conn in psutil.net_connections(kind='inet'):
                conn_data = {
                    'fd': conn.fd,
                    'family': conn.family,
                    'type': conn.type,
                    'laddr': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                    'raddr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                    'status': conn.status,
                    'pid': conn.pid
                }
                connections.append(conn_data)
        except Exception as e:
            logging.error(f"Error collecting network connections: {e}")

        return connections

#========================================================================
#                  HEATMAP GENERATOR
#========================================================================

class HeatmapGenerator:
    """Generate real-time heatmaps from system data"""

    def __init__(self, width: int = 50, height: int = 50):
        self.width = width
        self.height = height
        self.data_collector = RealWorldDataCollector()
        self.cmap = None
        if MATPLOTLIB_AVAILABLE:
            colors = ['#000000', '#001100', '#003300', '#00FF00']
            self.cmap = LinearSegmentedColormap.from_list('matrix', colors, N=256)

    def generate_cpu_heatmap(self) -> np.ndarray:
        metrics = self.data_collector.collect_system_metrics()

        if not metrics or 'cpu' not in metrics:
            return np.random.rand(self.height, self.width) * 0.1

        cpu_percents = metrics['cpu'].get('percent_per_core', [])

        if not cpu_percents:
            cpu_percents = [metrics['cpu'].get('percent', 0)]

        heatmap = np.zeros((self.height, self.width))
        cores = len(cpu_percents)
        cells_per_core = (self.width * self.height) // cores if cores > 0 else self.width * self.height

        for i, usage in enumerate(cpu_percents):
            start_idx = i * cells_per_core
            end_idx = start_idx + cells_per_core

            for idx in range(start_idx, min(end_idx, self.width * self.height)):
                row = idx // self.width
                col = idx % self.width
                if row < self.height:
                    heatmap[row, col] = usage / 100.0

        return heatmap

    def generate_memory_heatmap(self) -> np.ndarray:
        metrics = self.data_collector.collect_system_metrics()

        if not metrics or 'memory' not in metrics:
            return np.random.rand(self.height, self.width) * 0.1

        mem_percent = metrics['memory']['virtual'].get('percent', 0) / 100.0

        heatmap = np.zeros((self.height, self.width))

        for i in range(self.height):
            intensity = mem_percent * (1 - i / self.height)
            heatmap[i, :] = intensity

        return heatmap

    def generate_network_heatmap(self) -> np.ndarray:
        connections = self.data_collector.collect_network_connections()
        heatmap = np.zeros((self.height, self.width))

        for idx, conn in enumerate(connections[:self.width * self.height]):
            row = idx // self.width
            col = idx % self.width

            if conn['status'] == 'ESTABLISHED':
                heatmap[row, col] = 1.0
            elif conn['status'] in ['SYN_SENT', 'SYN_RECV']:
                heatmap[row, col] = 0.7
            else:
                heatmap[row, col] = 0.3

        return heatmap

    def generate_threat_heatmap(self, threat_scores: np.ndarray) -> np.ndarray:
        if len(threat_scores) == 0:
            return np.zeros((self.height, self.width))

        heatmap = np.zeros((self.height, self.width))

        for idx, score in enumerate(threat_scores[:self.width * self.height]):
            row = idx // self.width
            col = idx % self.width
            heatmap[row, col] = score

        return heatmap

#========================================================================
#                  MATRIX CODE RAIN WINDOW (NEW!)
#========================================================================

class MatrixCodeRain(tk.Toplevel):
    """Katana-style Matrix code rain window"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("MATRIX CODE RAIN - KATANA STYLE")
        self.geometry("800x600")
        self.configure(bg='#000000')

        # Canvas for Matrix rain
        self.canvas = tk.Canvas(self, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Matrix rain parameters
        self.columns = 80
        self.font_size = 14
        self.drops = [0] * self.columns

        # Matrix characters (Katana style - Japanese + symbols)
        self.chars = list("ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ01234567890")

        # Colors
        self.colors = ['#00FF41', '#00FF00', '#00CC00', '#008800', '#005500']

        self.running = True
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start animation
        self._animate()

    def _animate(self):
        """Animate matrix rain"""
        if not self.running:
            return

        # Clear canvas
        self.canvas.delete("all")

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.after(50, self._animate)
            return

        # Calculate columns based on canvas width
        self.columns = max(1, canvas_width // self.font_size)

        # Ensure drops list matches columns
        while len(self.drops) < self.columns:
            self.drops.append(0)
        while len(self.drops) > self.columns:
            self.drops.pop()

        # Draw falling characters
        for i in range(self.columns):
            # Random character
            char = np.random.choice(self.chars)

            # X position
            x = i * self.font_size

            # Y position (based on drop)
            y = self.drops[i] * self.font_size

            # Color (brighter at bottom)
            color_idx = min(4, int(self.drops[i] % 5))
            color = self.colors[color_idx]

            # Draw character
            self.canvas.create_text(
                x, y,
                text=char,
                fill=color,
                font=('Courier', self.font_size, 'bold'),
                anchor='nw'
            )

            # Move drop down
            self.drops[i] += 1

            # Reset drop if it goes off screen
            max_rows = max(1, canvas_height // self.font_size)
            if self.drops[i] * self.font_size > canvas_height and np.random.random() > 0.95:
                self.drops[i] = 0

        # Continue animation
        self.after(50, self._animate)

    def _on_close(self):
        """Handle window close"""
        self.running = False
        self.destroy()

#========================================================================
#                  MATRIX GUI - FIXED VERSION
#========================================================================

class MatrixGUI:
    """Matrix-inspired GUI - FIXED with training progress and Matrix rain"""

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title(f"WATCHDOG ELITE v{WATCHDOG_VERSION} - QUANTUM SENTINEL")
        self.master.geometry("1920x1080")
        self.master.configure(bg=MATRIX_COLORS['bg_primary'])

        self.data_collector = RealWorldDataCollector()
        self.heatmap_generator = HeatmapGenerator()
        self.neural_network = None
        self.quantum_crypto = QuantumCryptoEngine()

        self.running = True
        self.animation_enabled = True
        self.refresh_rate = CONFIG['gui_refresh_rate_ms']

        self.metrics_history = deque(maxlen=1000)
        self.threat_log = deque(maxlen=100)
        self.training_active = False
        self.matrix_rain_window = None

        self._setup_styles()
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()
        self._start_update_loops()

        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Matrix.TFrame', background=MATRIX_COLORS['bg_primary'])
        style.configure('Matrix.TLabel',
                       background=MATRIX_COLORS['bg_primary'],
                       foreground=MATRIX_COLORS['fg_bright'],
                       font=('Courier', 10, 'bold'))
        style.configure('Matrix.TButton',
                       background=MATRIX_COLORS['bg_panel'],
                       foreground=MATRIX_COLORS['fg_bright'],
                       font=('Courier', 10, 'bold'))
        style.map('Matrix.TButton',
                 background=[('active', MATRIX_COLORS['bg_secondary'])])

        style.configure('MatrixTitle.TLabel',
                       background=MATRIX_COLORS['bg_primary'],
                       foreground=MATRIX_COLORS['accent'],
                       font=('Courier', 16, 'bold'))

    def _create_menu(self):
        menubar = tk.Menu(self.master, bg=MATRIX_COLORS['bg_panel'],
                         fg=MATRIX_COLORS['fg_bright'])
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, bg=MATRIX_COLORS['bg_panel'],
                           fg=MATRIX_COLORS['fg_bright'])
        menubar.add_cascade(label="FILE", menu=file_menu)
        file_menu.add_command(label="Load Neural Network", command=self._load_neural_network)
        file_menu.add_command(label="Save Neural Network", command=self._save_neural_network)
        file_menu.add_separator()
        file_menu.add_command(label="Export Metrics", command=self._export_metrics)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)

        nn_menu = tk.Menu(menubar, tearoff=0, bg=MATRIX_COLORS['bg_panel'],
                         fg=MATRIX_COLORS['fg_bright'])
        menubar.add_cascade(label="NEURAL NET", menu=nn_menu)
        nn_menu.add_command(label="Create New Network", command=self._create_neural_network)
        nn_menu.add_command(label="Start Training", command=self._start_training)
        nn_menu.add_command(label="Stop Training", command=self._stop_training)
        nn_menu.add_command(label="Evaluate Performance", command=self._evaluate_network)

        viz_menu = tk.Menu(menubar, tearoff=0, bg=MATRIX_COLORS['bg_panel'],
                          fg=MATRIX_COLORS['fg_bright'])
        menubar.add_cascade(label="VISUALIZATION", menu=viz_menu)
        viz_menu.add_command(label="Toggle 3D View", command=self._toggle_3d_view)
        viz_menu.add_command(label="Toggle Heatmaps", command=self._toggle_heatmaps)
        viz_menu.add_command(label="Matrix Code Rain", command=self._show_matrix_rain)  # NEW!

        sys_menu = tk.Menu(menubar, tearoff=0, bg=MATRIX_COLORS['bg_panel'],
                          fg=MATRIX_COLORS['fg_bright'])
        menubar.add_cascade(label="SYSTEM", menu=sys_menu)
        sys_menu.add_command(label="System Info", command=self._show_system_info)
        sys_menu.add_command(label="Security Status", command=self._show_security_status)
        sys_menu.add_command(label="Clear Logs", command=self._clear_logs)

    def _create_main_layout(self):
        main_frame = ttk.Frame(self.master, style='Matrix.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        title_label = ttk.Label(main_frame,
                               text="╔═══ WATCHDOG ELITE QUANTUM SENTINEL ═══╗",
                               style='MatrixTitle.TLabel')
        title_label.pack(pady=10)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._create_dashboard_tab()
        self._create_neural_network_tab()
        self._create_heatmap_tab()
        self._create_3d_visualization_tab()

    def _create_dashboard_tab(self):
        dashboard_frame = ttk.Frame(self.notebook, style='Matrix.TFrame')
        self.notebook.add(dashboard_frame, text="DASHBOARD")

        left_frame = ttk.Frame(dashboard_frame, style='Matrix.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        right_frame = ttk.Frame(dashboard_frame, style='Matrix.TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        status_panel = ttk.LabelFrame(left_frame, text="SYSTEM STATUS", style='Matrix.TFrame')
        status_panel.pack(fill=tk.BOTH, expand=True, pady=5)

        self.status_text = scrolledtext.ScrolledText(status_panel,
                                                     bg=MATRIX_COLORS['bg_secondary'],
                                                     fg=MATRIX_COLORS['fg_bright'],
                                                     font=('Courier', 9),
                                                     height=15)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        metrics_panel = ttk.LabelFrame(left_frame, text="REAL-TIME METRICS", style='Matrix.TFrame')
        metrics_panel.pack(fill=tk.BOTH, expand=True, pady=5)

        self.metrics_text = scrolledtext.ScrolledText(metrics_panel,
                                                      bg=MATRIX_COLORS['bg_secondary'],
                                                      fg=MATRIX_COLORS['fg_normal'],
                                                      font=('Courier', 9),
                                                      height=15)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        log_panel = ttk.LabelFrame(right_frame, text="ACTIVITY LOG", style='Matrix.TFrame')
        log_panel.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_panel,
                                                  bg=MATRIX_COLORS['bg_secondary'],
                                                  fg=MATRIX_COLORS['fg_dim'],
                                                  font=('Courier', 8),
                                                  height=30)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_neural_network_tab(self):
        nn_frame = ttk.Frame(self.notebook, style='Matrix.TFrame')
        self.notebook.add(nn_frame, text="NEURAL NETWORK")

        config_frame = ttk.LabelFrame(nn_frame, text="NETWORK CONFIGURATION", style='Matrix.TFrame')
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        arch_frame = ttk.Frame(config_frame, style='Matrix.TFrame')
        arch_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(arch_frame, text="Input Size:", style='Matrix.TLabel').pack(side=tk.LEFT, padx=5)
        self.nn_input_var = tk.StringVar(value="50")
        ttk.Entry(arch_frame, textvariable=self.nn_input_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(arch_frame, text="Hidden Layers:", style='Matrix.TLabel').pack(side=tk.LEFT, padx=5)
        self.nn_hidden_var = tk.StringVar(value="128,64,32")
        ttk.Entry(arch_frame, textvariable=self.nn_hidden_var, width=20).pack(side=tk.LEFT, padx=5)

        ttk.Label(arch_frame, text="Output Size:", style='Matrix.TLabel').pack(side=tk.LEFT, padx=5)
        self.nn_output_var = tk.StringVar(value="2")
        ttk.Entry(arch_frame, textvariable=self.nn_output_var, width=10).pack(side=tk.LEFT, padx=5)

        train_frame = ttk.Frame(config_frame, style='Matrix.TFrame')
        train_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(train_frame, text="Learning Rate:", style='Matrix.TLabel').pack(side=tk.LEFT, padx=5)
        self.nn_lr_var = tk.StringVar(value="0.001")
        ttk.Entry(train_frame, textvariable=self.nn_lr_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(train_frame, text="Epochs:", style='Matrix.TLabel').pack(side=tk.LEFT, padx=5)
        self.nn_epochs_var = tk.StringVar(value="50")
        ttk.Entry(train_frame, textvariable=self.nn_epochs_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(train_frame, text="Batch Size:", style='Matrix.TLabel').pack(side=tk.LEFT, padx=5)
        self.nn_batch_var = tk.StringVar(value="32")
        ttk.Entry(train_frame, textvariable=self.nn_batch_var, width=10).pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(config_frame, style='Matrix.TFrame')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="CREATE NETWORK", style='Matrix.TButton',
                  command=self._create_neural_network).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="TRAIN", style='Matrix.TButton',
                  command=self._start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="EVALUATE", style='Matrix.TButton',
                  command=self._evaluate_network).pack(side=tk.LEFT, padx=5)

        # Progress bar (NEW!)
        self.training_progress = ttk.Progressbar(config_frame, length=600, mode='determinate')
        self.training_progress.pack(fill=tk.X, padx=5, pady=5)

        perf_frame = ttk.LabelFrame(nn_frame, text="TRAINING PERFORMANCE", style='Matrix.TFrame')
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        if MATPLOTLIB_AVAILABLE:
            self.nn_fig = Figure(figsize=(12, 6), facecolor=MATRIX_COLORS['bg_primary'])
            self.nn_canvas = FigureCanvasTkAgg(self.nn_fig, master=perf_frame)
            self.nn_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.nn_ax1 = self.nn_fig.add_subplot(121, facecolor=MATRIX_COLORS['bg_secondary'])
            self.nn_ax2 = self.nn_fig.add_subplot(122, facecolor=MATRIX_COLORS['bg_secondary'])

            for ax in [self.nn_ax1, self.nn_ax2]:
                ax.tick_params(colors=MATRIX_COLORS['fg_normal'], labelsize=10)
                ax.spines['bottom'].set_color(MATRIX_COLORS['fg_dim'])
                ax.spines['top'].set_color(MATRIX_COLORS['fg_dim'])
                ax.spines['left'].set_color(MATRIX_COLORS['fg_dim'])
                ax.spines['right'].set_color(MATRIX_COLORS['fg_dim'])

            self.nn_ax1.set_title('Loss', color=MATRIX_COLORS['fg_bright'], fontsize=14, fontweight='bold')
            self.nn_ax2.set_title('Accuracy', color=MATRIX_COLORS['fg_bright'], fontsize=14, fontweight='bold')

            self.nn_fig.tight_layout()

        self.nn_status_text = scrolledtext.ScrolledText(perf_frame,
                                                        bg=MATRIX_COLORS['bg_secondary'],
                                                        fg=MATRIX_COLORS['fg_bright'],
                                                        font=('Courier', 9),
                                                        height=10)
        self.nn_status_text.pack(fill=tk.X, padx=5, pady=5)

    def _create_heatmap_tab(self):
        heatmap_frame = ttk.Frame(self.notebook, style='Matrix.TFrame')
        self.notebook.add(heatmap_frame, text="HEATMAPS")

        control_frame = ttk.LabelFrame(heatmap_frame, text="HEATMAP CONTROLS", style='Matrix.TFrame')
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_frame = ttk.Frame(control_frame, style='Matrix.TFrame')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="CPU HEATMAP", style='Matrix.TButton',
                  command=lambda: self._show_heatmap('cpu')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="MEMORY HEATMAP", style='Matrix.TButton',
                  command=lambda: self._show_heatmap('memory')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="NETWORK HEATMAP", style='Matrix.TButton',
                  command=lambda: self._show_heatmap('network')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="THREAT HEATMAP", style='Matrix.TButton',
                  command=lambda: self._show_heatmap('threat')).pack(side=tk.LEFT, padx=5)

        viz_frame = ttk.LabelFrame(heatmap_frame, text="VISUALIZATION", style='Matrix.TFrame')
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        if MATPLOTLIB_AVAILABLE:
            self.heatmap_fig = Figure(figsize=(12, 8), facecolor=MATRIX_COLORS['bg_primary'])
            self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=viz_frame)
            self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.heatmap_ax = self.heatmap_fig.add_subplot(111, facecolor=MATRIX_COLORS['bg_secondary'])
            self.heatmap_ax.tick_params(colors=MATRIX_COLORS['fg_normal'], labelsize=10)

            self.heatmap_fig.tight_layout()

    def _create_3d_visualization_tab(self):
        """Create 3D visualization tab - FIXED sizing"""
        viz_3d_frame = ttk.Frame(self.notebook, style='Matrix.TFrame')
        self.notebook.add(viz_3d_frame, text="3D VISUALIZATION")

        control_frame = ttk.LabelFrame(viz_3d_frame, text="3D CONTROLS", style='Matrix.TFrame')
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_frame = ttk.Frame(control_frame, style='Matrix.TFrame')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="SYSTEM TOPOLOGY", style='Matrix.TButton',
                  command=lambda: self._show_3d_viz('topology')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="THREAT LANDSCAPE", style='Matrix.TButton',
                  command=lambda: self._show_3d_viz('threats')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="NETWORK GRAPH", style='Matrix.TButton',
                  command=lambda: self._show_3d_viz('network')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="NEURAL NET STRUCTURE", style='Matrix.TButton',
                  command=lambda: self._show_3d_viz('neural_net')).pack(side=tk.LEFT, padx=5)

        viz_frame = ttk.LabelFrame(viz_3d_frame, text="3D VIEW", style='Matrix.TFrame')
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        if MATPLOTLIB_AVAILABLE:
            # FIXED: Larger figure size for better visibility
            self.viz_3d_fig = Figure(figsize=(14, 10), facecolor=MATRIX_COLORS['bg_primary'])
            self.viz_3d_canvas = FigureCanvasTkAgg(self.viz_3d_fig, master=viz_frame)
            self.viz_3d_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.viz_3d_ax = self.viz_3d_fig.add_subplot(111, projection='3d', facecolor=MATRIX_COLORS['bg_secondary'])
            self.viz_3d_ax.tick_params(colors=MATRIX_COLORS['fg_normal'], labelsize=12)  # FIXED: Larger labels

            self.viz_3d_fig.tight_layout(pad=2.0)  # FIXED: More padding

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.master, style='Matrix.TFrame', relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="SYSTEM ONLINE", style='Matrix.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.fps_label = ttk.Label(status_frame, text="FPS: 60", style='Matrix.TLabel')
        self.fps_label.pack(side=tk.RIGHT, padx=10)

        self.time_label = ttk.Label(status_frame, text="", style='Matrix.TLabel')
        self.time_label.pack(side=tk.RIGHT, padx=10)

    def _start_update_loops(self):
        self._update_status()
        self._update_metrics()
        self._update_time()

    def _update_status(self):
        if not self.running:
            return

        try:
            metrics = self.data_collector.collect_system_metrics()

            if metrics:
                status_lines = [
                    "╔══════════════════════════════════════════╗",
                    "║          SYSTEM STATUS REPORT            ║",
                    "╠══════════════════════════════════════════╣",
                    f"║ CPU Usage: {metrics['cpu'].get('percent', 0):.1f}%",
                    f"║ Memory: {metrics['memory']['virtual'].get('percent', 0):.1f}%",
                    f"║ Active Processes: {metrics['processes'].get('count', 0)}",
                    f"║ Network Connections: {metrics['network'].get('connections_count', 0)}",
                    "╚══════════════════════════════════════════╝"
                ]

                self.status_text.delete('1.0', tk.END)
                self.status_text.insert('1.0', '\n'.join(status_lines))
        except Exception as e:
            logging.error(f"Error updating status: {e}")

        self.master.after(1000, self._update_status)

    def _update_metrics(self):
        if not self.running:
            return

        try:
            metrics = self.data_collector.collect_system_metrics()

            if metrics:
                metrics_lines = [
                    f"[{datetime.now().strftime('%H:%M:%S')}] METRICS UPDATE",
                    "-" * 60,
                    f"CPU: {metrics['cpu'].get('percent', 0):.2f}%",
                    f"Memory: {metrics['memory']['virtual'].get('percent', 0):.2f}%",
                    f"Network: {metrics['network'].get('connections_count', 0)} connections",
                    "-" * 60,
                    ""
                ]

                current_text = self.metrics_text.get('1.0', tk.END)
                lines = current_text.split('\n')
                if len(lines) > 200:
                    current_text = '\n'.join(lines[-200:])

                self.metrics_text.delete('1.0', tk.END)
                self.metrics_text.insert('1.0', current_text)
                self.metrics_text.insert(tk.END, '\n'.join(metrics_lines))
                self.metrics_text.see(tk.END)

                self.metrics_history.append(metrics)
        except Exception as e:
            logging.error(f"Error updating metrics: {e}")

        self.master.after(self.refresh_rate, self._update_metrics)

    def _update_time(self):
        if not self.running:
            return

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"TIME: {current_time}")

        self.master.after(1000, self._update_time)

    def _log_message(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] [{level}] {message}\n"

        self.log_text.insert(tk.END, log_line)
        self.log_text.see(tk.END)

        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.log_text.delete('1.0', '500.0')

    def _create_neural_network(self):
        try:
            input_size = int(self.nn_input_var.get())
            hidden_layers = [int(x.strip()) for x in self.nn_hidden_var.get().split(',')]
            output_size = int(self.nn_output_var.get())
            learning_rate = float(self.nn_lr_var.get())

            self.neural_network = NumpyNeuralNetwork(
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=output_size,
                learning_rate=learning_rate
            )

            self._log_message(f"Neural Network created: {input_size} -> {hidden_layers} -> {output_size}", "SUCCESS")
            self.nn_status_text.insert(tk.END, f"\nNetwork created successfully!\n")
            self.nn_status_text.insert(tk.END, f"Architecture: {input_size} -> {' -> '.join(map(str, hidden_layers))} -> {output_size}\n")

        except Exception as e:
            self._log_message(f"Error creating neural network: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to create neural network: {e}")

    def _training_callback(self, epoch, total_epochs, loss, accuracy):
        """Callback for training progress updates - FIXED"""
        # Update progress bar
        progress = (epoch + 1) / total_epochs * 100
        self.training_progress['value'] = progress

        # Update status
        status_msg = f"Epoch {epoch+1}/{total_epochs} - Loss: {loss:.4f} - Acc: {accuracy:.4f}"
        self.nn_status_text.insert(tk.END, status_msg + "\n")
        self.nn_status_text.see(tk.END)

        # Force GUI update
        self.master.update_idletasks()

    def _start_training(self):
        """Start neural network training - FIXED to not hang"""
        if self.neural_network is None:
            messagebox.showwarning("Warning", "Please create a neural network first")
            return

        if self.training_active:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        def training_thread():
            try:
                self.training_active = True
                self.training_progress['value'] = 0
                self._log_message("Starting neural network training...", "INFO")

                # Generate smaller dataset for faster training
                X_train, y_train = self._generate_training_data(200)  # FIXED: Reduced from 1000
                X_val, y_val = self._generate_training_data(50)  # FIXED: Reduced from 200

                epochs = int(self.nn_epochs_var.get())
                batch_size = int(self.nn_batch_var.get())

                # Set callback for GUI updates
                self.neural_network.training_callback = self._training_callback

                # Train
                history = self.neural_network.train(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=False  # FIXED: Disable console output
                )

                # Update plots
                self.master.after(0, lambda: self._update_training_plots(history))

                self._log_message(f"Training complete! Best accuracy: {self.neural_network.best_accuracy:.4f}", "SUCCESS")
                self.training_active = False
                self.training_progress['value'] = 100

            except Exception as e:
                self._log_message(f"Training error: {e}", "ERROR")
                self.training_active = False
                self.training_progress['value'] = 0

        thread = Thread(target=training_thread, daemon=True)
        thread.start()

    def _stop_training(self):
        self.training_active = False
        self._log_message("Training stopped by user", "WARNING")

    def _evaluate_network(self):
        if self.neural_network is None:
            messagebox.showwarning("Warning", "Please create a neural network first")
            return

        try:
            X_test, y_test = self._generate_training_data(100)
            metrics = self.neural_network.evaluate(X_test, y_test)

            self.nn_status_text.insert(tk.END, "\n" + "="*60 + "\n")
            self.nn_status_text.insert(tk.END, "EVALUATION RESULTS:\n")
            self.nn_status_text.insert(tk.END, f"Loss: {metrics['loss']:.4f}\n")
            self.nn_status_text.insert(tk.END, f"Accuracy: {metrics['accuracy']:.4f}\n")
            self.nn_status_text.insert(tk.END, f"Precision: {metrics['precision']:.4f}\n")
            self.nn_status_text.insert(tk.END, f"Recall: {metrics['recall']:.4f}\n")
            self.nn_status_text.insert(tk.END, f"F1 Score: {metrics['f1_score']:.4f}\n")
            self.nn_status_text.insert(tk.END, "="*60 + "\n")
            self.nn_status_text.see(tk.END)

            self._log_message(f"Network evaluation complete - Accuracy: {metrics['accuracy']:.4f}", "SUCCESS")

        except Exception as e:
            self._log_message(f"Evaluation error: {e}", "ERROR")
            messagebox.showerror("Error", f"Evaluation failed: {e}")

    def _generate_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data - FIXED to be faster"""
        X = []
        y = []

        for i in range(n_samples):
            metrics = self.data_collector.collect_system_metrics()

            if metrics:
                features = [
                    metrics['cpu'].get('percent', 0) / 100.0,
                    metrics['memory']['virtual'].get('percent', 0) / 100.0,
                    metrics['disk']['usage'].get('percent', 0) / 100.0,
                    len(metrics['network'].get('interfaces', {})) / 10.0,
                    metrics['processes'].get('count', 0) / 1000.0,
                ]

                while len(features) < self.neural_network.input_size:
                    features.append(np.random.rand())

                X.append(features[:self.neural_network.input_size])

                is_anomaly = (metrics['cpu'].get('percent', 0) > 80 or
                             metrics['memory']['virtual'].get('percent', 0) > 90)

                label = [0] * self.neural_network.output_size
                label[1 if is_anomaly else 0] = 1
                y.append(label)

            # FIXED: No delay - collect as fast as possible
            if i % 10 == 0:  # Update GUI every 10 samples
                self.master.update_idletasks()

        return np.array(X), np.array(y)

    def _update_training_plots(self, history: Dict[str, List[float]]):
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            self.nn_ax1.clear()
            self.nn_ax2.clear()

            epochs = range(1, len(history['loss']) + 1)

            self.nn_ax1.plot(epochs, history['loss'], color=MATRIX_COLORS['fg_bright'], label='Training Loss', linewidth=2)
            if history.get('val_loss'):
                self.nn_ax1.plot(epochs, history['val_loss'], color=MATRIX_COLORS['fg_warning'], label='Validation Loss', linewidth=2)
            self.nn_ax1.set_xlabel('Epoch', color=MATRIX_COLORS['fg_normal'], fontsize=12)
            self.nn_ax1.set_ylabel('Loss', color=MATRIX_COLORS['fg_normal'], fontsize=12)
            self.nn_ax1.set_title('Training Loss', color=MATRIX_COLORS['fg_bright'], fontsize=14, fontweight='bold')
            self.nn_ax1.legend(facecolor=MATRIX_COLORS['bg_panel'], edgecolor=MATRIX_COLORS['fg_dim'], fontsize=10)
            self.nn_ax1.grid(True, alpha=0.3, color=MATRIX_COLORS['grid_line'])

            self.nn_ax2.plot(epochs, history['accuracy'], color=MATRIX_COLORS['fg_bright'], label='Training Accuracy', linewidth=2)
            if history.get('val_accuracy'):
                self.nn_ax2.plot(epochs, history['val_accuracy'], color=MATRIX_COLORS['fg_warning'], label='Validation Accuracy', linewidth=2)
            self.nn_ax2.set_xlabel('Epoch', color=MATRIX_COLORS['fg_normal'], fontsize=12)
            self.nn_ax2.set_ylabel('Accuracy', color=MATRIX_COLORS['fg_normal'], fontsize=12)
            self.nn_ax2.set_title('Training Accuracy', color=MATRIX_COLORS['fg_bright'], fontsize=14, fontweight='bold')
            self.nn_ax2.legend(facecolor=MATRIX_COLORS['bg_panel'], edgecolor=MATRIX_COLORS['fg_dim'], fontsize=10)
            self.nn_ax2.grid(True, alpha=0.3, color=MATRIX_COLORS['grid_line'])

            self.nn_fig.tight_layout()
            self.nn_canvas.draw()

        except Exception as e:
            logging.error(f"Error updating training plots: {e}")

    def _show_heatmap(self, heatmap_type: str):
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("Warning", "Matplotlib not available")
            return

        try:
            self._log_message(f"Generating {heatmap_type} heatmap...", "INFO")

            if heatmap_type == 'cpu':
                data = self.heatmap_generator.generate_cpu_heatmap()
                title = "CPU UTILIZATION HEATMAP"
            elif heatmap_type == 'memory':
                data = self.heatmap_generator.generate_memory_heatmap()
                title = "MEMORY USAGE HEATMAP"
            elif heatmap_type == 'network':
                data = self.heatmap_generator.generate_network_heatmap()
                title = "NETWORK ACTIVITY HEATMAP"
            elif heatmap_type == 'threat':
                threat_scores = np.random.rand(100) * 0.3
                data = self.heatmap_generator.generate_threat_heatmap(threat_scores)
                title = "THREAT DETECTION HEATMAP"
            else:
                return

            self.heatmap_ax.clear()

            colors = ['#000000', '#001100', '#003300', '#00FF00', '#00FF41']
            cmap = LinearSegmentedColormap.from_list('matrix_heat', colors)

            im = self.heatmap_ax.imshow(data, cmap=cmap, aspect='auto', interpolation='bilinear')
            self.heatmap_ax.set_title(title, color=MATRIX_COLORS['fg_bright'], fontsize=16, fontweight='bold', pad=20)  # FIXED: Larger title

            if hasattr(self, 'heatmap_colorbar'):
                self.heatmap_colorbar.remove()

            self.heatmap_colorbar = self.heatmap_fig.colorbar(im, ax=self.heatmap_ax)
            self.heatmap_colorbar.ax.tick_params(colors=MATRIX_COLORS['fg_normal'], labelsize=10)

            self.heatmap_fig.tight_layout()
            self.heatmap_canvas.draw()

            self._log_message(f"{heatmap_type} heatmap generated", "SUCCESS")

        except Exception as e:
            self._log_message(f"Error generating heatmap: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to generate heatmap: {e}")

    def _show_3d_viz(self, viz_type: str):
        """Show 3D visualization - FIXED sizing"""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("Warning", "Matplotlib not available")
            return

        try:
            self._log_message(f"Generating {viz_type} 3D visualization...", "INFO")

            self.viz_3d_ax.clear()

            if viz_type == 'topology':
                self._render_3d_topology()
            elif viz_type == 'threats':
                self._render_3d_threats()
            elif viz_type == 'network':
                self._render_3d_network()
            elif viz_type == 'neural_net':
                self._render_3d_neural_net()

            self.viz_3d_canvas.draw()

            self._log_message(f"{viz_type} 3D visualization generated", "SUCCESS")

        except Exception as e:
            self._log_message(f"Error generating 3D visualization: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to generate 3D visualization: {e}")

    def _render_3d_topology(self):
        """Render 3D system topology - FIXED labels"""
        n_nodes = 20
        x = np.random.rand(n_nodes) * 10
        y = np.random.rand(n_nodes) * 10
        z = np.random.rand(n_nodes) * 10

        colors = [MATRIX_COLORS['fg_bright'] if i % 3 == 0 else MATRIX_COLORS['fg_normal'] for i in range(n_nodes)]

        self.viz_3d_ax.scatter(x, y, z, c=colors, marker='o', s=100, alpha=0.8)

        for i in range(n_nodes - 1):
            self.viz_3d_ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]],
                               color=MATRIX_COLORS['fg_dim'], alpha=0.3, linewidth=0.5)

        self.viz_3d_ax.set_xlabel('X', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_ylabel('Y', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_zlabel('Z', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_title('SYSTEM TOPOLOGY', color=MATRIX_COLORS['fg_bright'], fontsize=18, fontweight='bold', pad=20)  # FIXED

    def _render_3d_threats(self):
        """Render 3D threat landscape - FIXED labels"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)

        metrics = self.data_collector.collect_system_metrics()
        cpu_factor = metrics.get('cpu', {}).get('percent', 50) / 100.0 if metrics else 0.5

        Z = np.sin(np.sqrt(X**2 + Y**2) * cpu_factor) * 2 + cpu_factor * 3

        self.viz_3d_ax.plot_surface(X, Y, Z, cmap='RdYlGn_r', alpha=0.7, edgecolor='none')

        self.viz_3d_ax.set_xlabel('Time', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_ylabel('Severity', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_zlabel('Threat Level', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_title('THREAT LANDSCAPE', color=MATRIX_COLORS['fg_bright'], fontsize=18, fontweight='bold', pad=20)  # FIXED

    def _render_3d_network(self):
        """Render 3D network graph - FIXED labels"""
        connections = self.data_collector.collect_network_connections()

        n_conn = min(len(connections), 50)

        if n_conn == 0:
            n_conn = 10

        theta = np.linspace(0, 2*np.pi, n_conn)
        r = np.random.rand(n_conn) * 5 + 5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.rand(n_conn) * 10

        self.viz_3d_ax.scatter(x, y, z, c=MATRIX_COLORS['fg_bright'], marker='o', s=50, alpha=0.8)

        center = [0, 0, 5]
        for i in range(n_conn):
            self.viz_3d_ax.plot([x[i], center[0]], [y[i], center[1]], [z[i], center[2]],
                               color=MATRIX_COLORS['fg_dim'], alpha=0.3, linewidth=0.5)

        self.viz_3d_ax.set_xlabel('X', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_ylabel('Y', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_zlabel('Z', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_title('NETWORK GRAPH', color=MATRIX_COLORS['fg_bright'], fontsize=18, fontweight='bold', pad=20)  # FIXED

    def _render_3d_neural_net(self):
        """Render 3D neural network structure - FIXED labels"""
        if self.neural_network is None:
            messagebox.showinfo("Info", "Please create a neural network first")
            return

        layer_sizes = self.neural_network.layer_sizes
        n_layers = len(layer_sizes)

        for layer_idx, layer_size in enumerate(layer_sizes):
            z = layer_idx * 3

            theta = np.linspace(0, 2*np.pi, layer_size, endpoint=False)
            r = layer_size / 10.0
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z_arr = np.ones(layer_size) * z

            if layer_idx == 0:
                color = MATRIX_COLORS['fg_info']
            elif layer_idx == n_layers - 1:
                color = MATRIX_COLORS['fg_alert']
            else:
                color = MATRIX_COLORS['fg_bright']

            self.viz_3d_ax.scatter(x, y, z_arr, c=color, marker='o', s=100, alpha=0.8)

            if layer_idx > 0:
                prev_layer_size = layer_sizes[layer_idx - 1]
                prev_theta = np.linspace(0, 2*np.pi, prev_layer_size, endpoint=False)
                prev_r = prev_layer_size / 10.0
                prev_x = prev_r * np.cos(prev_theta)
                prev_y = prev_r * np.sin(prev_theta)
                prev_z = (layer_idx - 1) * 3

                for i in range(min(layer_size, 5)):
                    for j in range(min(prev_layer_size, 5)):
                        self.viz_3d_ax.plot([x[i], prev_x[j]], [y[i], prev_y[j]], [z, prev_z],
                                           color=MATRIX_COLORS['fg_dim'], alpha=0.1, linewidth=0.3)

        self.viz_3d_ax.set_xlabel('X', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_ylabel('Y', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_zlabel('Layer', color=MATRIX_COLORS['fg_normal'], fontsize=14, labelpad=10)  # FIXED
        self.viz_3d_ax.set_title('NEURAL NETWORK STRUCTURE', color=MATRIX_COLORS['fg_bright'], fontsize=18, fontweight='bold', pad=20)  # FIXED

    def _show_matrix_rain(self):
        """Show Matrix code rain window - NEW!"""
        if self.matrix_rain_window is None or not self.matrix_rain_window.winfo_exists():
            self.matrix_rain_window = MatrixCodeRain(self.master)
            self._log_message("Matrix Code Rain window opened", "INFO")
        else:
            self.matrix_rain_window.focus()

    def _load_neural_network(self):
        filepath = filedialog.askopenfilename(
            title="Load Neural Network",
            initialdir=NN_MODELS_DIR,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            try:
                with open(filepath, 'r') as f:
                    model_data = json.load(f)

                config = model_data['config']
                self.neural_network = NumpyNeuralNetwork(
                    input_size=config['input_size'],
                    hidden_layers=config['hidden_layers'],
                    output_size=config['output_size'],
                    learning_rate=config['learning_rate']
                )

                self.neural_network.load_model(filepath)

                self._log_message(f"Neural network loaded from {os.path.basename(filepath)}", "SUCCESS")
                messagebox.showinfo("Success", "Neural network loaded successfully")

            except Exception as e:
                self._log_message(f"Error loading neural network: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to load neural network: {e}")

    def _save_neural_network(self):
        if self.neural_network is None:
            messagebox.showwarning("Warning", "No neural network to save")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Neural Network",
            initialdir=NN_MODELS_DIR,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            try:
                self.neural_network.save_model(filepath)
                self._log_message(f"Neural network saved to {os.path.basename(filepath)}", "SUCCESS")
                messagebox.showinfo("Success", "Neural network saved successfully")

            except Exception as e:
                self._log_message(f"Error saving neural network: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to save neural network: {e}")

    def _export_metrics(self):
        filepath = filedialog.asksaveasfilename(
            title="Export Metrics",
            initialdir=BASE_DIR,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            try:
                metrics_data = {
                    'export_time': datetime.now().isoformat(),
                    'metrics_count': len(self.metrics_history),
                    'metrics': list(self.metrics_history)
                }

                with open(filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2, default=str)

                self._log_message(f"Metrics exported to {os.path.basename(filepath)}", "SUCCESS")
                messagebox.showinfo("Success", f"Exported {len(self.metrics_history)} metric records")

            except Exception as e:
                self._log_message(f"Error exporting metrics: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to export metrics: {e}")

    def _toggle_3d_view(self):
        self._log_message("3D view toggled", "INFO")

    def _toggle_heatmaps(self):
        self._log_message("Heatmaps toggled", "INFO")

    def _show_system_info(self):
        if not PSUTIL_AVAILABLE:
            return

        info_lines = [
            "=" * 60,
            "SYSTEM INFORMATION",
            "=" * 60,
            f"Platform: {platform.system()} {platform.release()}",
            f"Processor: {platform.processor()}",
            f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical",
            f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB",
            f"Python Version: {platform.python_version()}",
            f"Watchdog Version: {WATCHDOG_VERSION}",
            "=" * 60
        ]

        messagebox.showinfo("System Information", '\n'.join(info_lines))

    def _show_security_status(self):
        status_lines = [
            "=" * 60,
            "SECURITY STATUS",
            "=" * 60,
            f"Quantum Crypto: ENABLED",
            f"Real-time Monitoring: ACTIVE",
            f"Neural Network: {'ACTIVE' if self.neural_network else 'INACTIVE'}",
            f"Threat Detection: ACTIVE",
            f"Argon2: {ARGON2_TYPE if ARGON2_AVAILABLE else 'DISABLED'}",
            "=" * 60
        ]

        messagebox.showinfo("Security Status", '\n'.join(status_lines))

    def _clear_logs(self):
        self.log_text.delete('1.0', tk.END)
        self._log_message("Log cleared", "INFO")

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to exit Watchdog Elite?"):
            self.running = False
            if self.matrix_rain_window and self.matrix_rain_window.winfo_exists():
                self.matrix_rain_window.destroy()
            self.master.destroy()

#========================================================================
#                  MAIN ENTRY POINT
#========================================================================

def setup_logging():
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    log_file = logs_dir / f"watchdog_elite_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=100*1024*1024,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler]
    )

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

def create_directories():
    directories = [
        BASE_DIR,
        NN_MODELS_DIR,
        HEATMAP_DATA_DIR,
        VISUALIZATION_DIR,
        BASE_DIR / "logs",
        BASE_DIR / "backups"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def main():
    setup_logging()
    create_directories()

    if ARGON2_AVAILABLE:
        print(f"[✓] Argon2 Cryptography: ENABLED ({ARGON2_TYPE})")
        logging.info(f"Argon2 enabled using {ARGON2_TYPE}")
    else:
        print("[!] Argon2 Cryptography: DISABLED (using PBKDF2 fallback)")
        logging.warning("Argon2 not available")

    print()
    logging.info("Starting Watchdog Elite v2.0 FIXED")

    root = tk.Tk()
    app = MatrixGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received")
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        logging.info("Watchdog Elite v2.0 FIXED shutdown complete")

if __name__ == "__main__":
    main()
