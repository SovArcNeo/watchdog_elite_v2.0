# Watchdog Elite v2.0
[![Status](https://img.shields.io/badge/status-operational-success.svg)](https://github.com/YOUR_USERNAME/watchdog-elite) [![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20ChromeOS-blue.svg)](https://github.com/YOUR_USERNAME/watchdog-elite) [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md) 
A standalone, AI-powered "Quantum Sentinel" with a custom Matrix-style GUI, built from the ground up in Python. This platform provides a complete environment for neural network training, real-time system monitoring, and advanced threat visualization.
## 📋 Table of Contents - 
[About The Project](#-about-the-project)  
[Key Features](#-key-features)  
[Getting Started](#-getting-started) 
[Usage](#-usage)
[File Structure](#-file-structure)
[Testing](#-testing)
[License](#-license)
[Contact](#-contact)
## 🚀 About The Project **Watchdog Elite v2.0** is the production-ready evolution of a standalone security agent. It has been re-engineered from the ground up as a complete, multi-threaded, AI-powered cybersecurity platform with a single `watchdog_elite_v2.0.py` file (2,325 lines)
The core philosophy of this project is **sovereignty and performance**: * **No TensorFlow/PyTorch:** It features a complete neural network class built from scratch using **pure NumPy**. * **No Simulations:**
All data is collected from **real-world system metrics** via `psutil`. * **No Dependencies:** It's designed to run with a minimal footprint on Linux/ChromeOS. This platform is a showcase of what is possible with native Python libraries, featuring a custom Tkinter GUI, live 3D visualizations, and quantum-resistant cryptography.
✨ Key Features ### 1. Matrix-Inspired Sci-Fi GUI A multi-tabbed, high-performance interface built with Tkinter, optimized for 1920x1080 with a 60 FPS animation rate.
* **Dashboard:** Real-time system status, metrics, and activity log
* * **Neural Network:** Interactive training interface and live performance graphs.
* * **Heatmaps:** CPU, Memory, Network, and Threat visualizations.
* * **3D Visualization:** System topology, threat landscapes, and network graphs.
* * **Threat Detection:** Real-time threat scanning and alerts.
* * **System Metrics:** Detailed `psutil` system information.
* * Pure NumPy Neural Network A custom-built `NumpyNeuralNetwork` class with **zero TensorFlow dependencies**.
* * **Configurable Architecture:** Define custom hidden layers (e.g., `[128, 64, 32]`).
* * **Modern Features:** Includes ReLU, Leaky ReLU, Sigmoid, and Softmax activations, full forward/backward propagation, mini-batch training, dropout, and He initialization.
* * **Full Lifecycle:** Track training history and save/load models to JSON.
* * **Metrics:** Calculates accuracy, precision, recall, and F1-score. ```python # Example of the pure NumPy NN nn = NumpyNeuralNetwork( input_size=50, hidden_layers=[128, 64, 32], output_size=2, learning_rate=0.001 ) nn.train(X_train, y_train, epochs=100, batch_size=32) nn.save_model("nn_models/my_model.json") 

##Advanced Data Visualization

3D Visualizations: Live, interactive matplotlib 3D projections for:

System Topology

Threat Landscapes (3D surface plots)

Network Graphs

Neural Network Structure

Heatmaps: A HeatmapGenerator class for real-time, high-resolution (50x50 grid) displays of:

CPU Heatmap (per-core utilization)

Memory Heatmap (RAM gradients)

Network Heatmap (connection activity)

Threat Heatmap (threat score distribution)

##Real-World Data Collection (No Simulations)

The RealWorldDataCollector class uses psutil to pull live data from the host machine.

Metrics Collected: CPU (per core), Virtual/Swap Memory, Disk I/O, Network Connections (with PIDs), and detailed Process lists.

Performance: Features a 10,000-sample metrics buffer and thread-safe collection.

5. Quantum-Resistant Cryptography

A QuantumCryptoEngine for hardened security.

Key Derivation: Uses Argon2id (via argon2-cffi) with 256MB memory cost and 16-thread parallelism.

Multi-Layer Encryption: ChaCha20-Poly1305, AES-256-GCM, and more.

Signatures: Post-quantum resistant Ed448 signatures.

Entropy Sources: Gathers entropy from hardware RNG, CPU jitter, and network timing.

6. Machine Learning Ensemble

Integrates scikit-learn for rapid prototyping and ensemble anomaly detection, using models like:

Isolation Forest

DBSCAN

One-Class SVM

Local Outlier Factor (LOF)

Gradient Boosting Classifier

Live Neural Network Training:

3D Threat Landscape:

Threat Heatmap:


Usage

The system is controlled entirely through the GUI.

Main Menu:

File: Load/Save Neural Networks, Export Metrics, Exit.

Neural Net: Create new networks, Start/Stop training, and Evaluate.

Visualization: Toggle 3D views, Heatmaps, and animations.

Dashboard Tab:

START SCAN: Begins continuous real-time monitoring.

STOP SCAN: Pauses monitoring.

SNAPSHOT: Saves current metrics.

Neural Network Tab:

Use the input fields to define your network architecture (e.g., "128,64,32" for hidden layers).

Press TRAIN to begin training the pure NumPy network.

Watch the Training Loss and Training Accuracy graphs update in real-time.

📁 File Structure

The application creates a ~/.watchdog_elite/ directory in your home folder to store all data, ensuring the main project folder remains clean.

~/.watchdog_elite/
├── logs/ # System logs
├── ml_models/ # Sklearn models (joblib files)
├── nn_models/ # Pure NumPy Neural Network models (JSON files)
├── quantum_secure/ # Encrypted data
│  └── quantum_keys/ # Quantum keys
├── heatmaps/ # Saved heatmap data
├── visualizations/ # Exported visualizations
├── forensics/ # Forensic data logs
├── backups/ # System backups
├── state/ # Runtime state
└── cache/ # Temporary cache 

🧪 Testing

A test script is included to validate all major subsystems.

Bash

python3 test_v2_subsystems.py 

Expected Output:

[✔] NumPy Neural Network - ALL TESTS PASSED [✔] Quantum Cryptography Engine - ALL TESTS PASSED [✔] Real-World Data Collector - ALL TESTS PASSED [✔] Heatmap Generator - ALL TESTS PASSED 

📄 License

Distributed under the MIT License. See LICENSE.md for more information.

💬 Contact

SovArcNeo (The Architect) - [c_davenport795@proton.me]

This system is a proof-of-concept for a larger "sovereign AI" ecosystem. I am available for high-level AI/Security consultation and custom agent development.

## Commercial licensing / support / custom builds: DM me
