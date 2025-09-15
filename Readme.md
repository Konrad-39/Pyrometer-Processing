# Multi-Color Pyrometer Analysis Tool

A comprehensive GUI application for analyzing multi-color pyrometry data with advanced SNR analysis, temperature calculations, and uncertainty quantification.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

## 🔥 Features

### Pyrometry Analysis
- **3-Color and 32-Color Pyrometry**: Support for both standard 3-detector and advanced 32-detector systems
- **Ratio Pyrometry**: Wien's law-based temperature calculations using detector ratios
- **Multiple Calibration Points**: Support for multiple reference temperatures with uncertainty analysis
- **Background & Dark Correction**: Automatic signal correction for accurate measurements

### 📊 Advanced SNR Analysis
- **Rolling SNR Analysis**: Time-resolved signal-to-noise ratio calculations
- **Comprehensive Metrics**: RMS SNR, peak-to-peak noise, dynamic range analysis
- **Quality Assessment**: Automatic detector quality grading (Excellent/Good/Fair/Poor)
- **Threshold-Based Filtering**: Configurable SNR thresholds with percentage-time requirements

### 📈 Data Visualization
- **Real-time Plotting**: Interactive matplotlib integration
- **Temperature vs Time**: Multi-calibration results with uncertainty bands
- **Signal Analysis**: Detector signals, ratios, and corrections
- **SNR Monitoring**: Rolling SNR visualization over time

### 💾 Data Management
- **Flexible File Loading**: Automatic detection of detector files (C1, C2, C3... format)
- **Batch Processing**: Load and average multiple files per detector
- **Data Export**: CSV export with metadata and analysis parameters
- **Results Saving**: High-resolution plot export (PNG, PDF, SVG)

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pyrometer-analysis-tool.git
   cd pyrometer-analysis-tool

pip install -r requirements.txt

tkinter
matplotlib>=3.5.0
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0

File Organization

project_folder/
├── experimental/
│   ├── C1_experiment_001.txt
│   ├── C2_experiment_001.txt
│   └── C3_experiment_001.txt
├── calibration/
│   ├── C1_calibration_001.txt
│   ├── C2_calibration_001.txt
│   └── C3_calibration_001.txt
├── background/
│   ├── C1_background_001.txt
│   ├── C2_background_001.txt
│   └── C3_background_001.txt
└── dark/
    ├── C1_dark_001.txt
    ├── C2_dark_001.txt
    └── C3_dark_001.txt


