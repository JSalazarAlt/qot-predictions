# QoT Predictions in Optical Networks - Network and Data Analysis Project

A machine learning project implementing **Quality of Transmission (QoT) prediction** in optical networks using **Gradient Boosting** algorithms for SNR estimation and Modulation Format assignment.

## 📋 Project Overview

This project addresses the challenge of predicting Signal-to-Noise Ratio (SNR) in optical networks to enable optimal Modulation Format (MF) assignment. The implementation compares two approaches:

1. **Regression-based approach**: Predict SNR values, then map to Modulation Formats
2. **Classification-based approach**: Directly predict Modulation Formats

### 🎯 Problem Statement

In optical network planning, we must assign Modulation Formats before launching signals into the network. SNR varies across different paths due to:
- Varying numbers of optical amplifiers
- Different levels of interfering channels
- Path-specific characteristics

**Machine Learning Advantage**: High accuracy with imprecisely known path parameters and low margins (better network resource utilization).

## 🔬 Technical Approach

### Dataset Characteristics
- **Networks**: 17-node German and 19-node European optical networks
- **Data Type**: Simulated dataset with analytical interference modeling
- **Structure**: Path features + Interference features → SNR labels

### Feature Engineering
**Path Features:**
- Length (Km) of fiber spans (span 1 to span N)

**Interference Features:**
- Number of channels in each link (link 1 to link M)

**Target Variable:**
- SNR (dB) - Signal-to-Noise Ratio

### Machine Learning Models

#### 1. Probabilistic Regression (Quantile Regression)
- **Algorithm**: Gradient Boosting Regressor with quantile loss
- **Advantage**: Provides prediction intervals and uncertainty quantification
- **Implementation**: `GradientBoostingRegressor(loss='quantile', alpha=α)`
- **Quantile Control**: 
  - High quantiles (α→1.0): Less conservative, penalizes underestimations
  - Low quantiles (α→0.0): More conservative, penalizes overestimations

#### 2. Multi-class Classification
- **Algorithm**: Gradient Boosting Classifier
- **Target**: Direct MF prediction (QPSK, 8QAM, 16QAM, 32QAM, 64QAM)
- **Implementation**: `GradientBoostingClassifier`

### Modulation Format Mapping
| MF | Required SNR (dB) |
|----|-------------------|
| QPSK | 8.7 |
| 8QAM | 12.8 |
| 16QAM | 15.2 |
| 32QAM | 18.2 |
| 64QAM | 21.0 |

## 📁 Project Structure

```
qot-predictions/
├── README.md
├── requirements.txt
├── .gitignore
├── qot_predictions_optical_networks.py          # Main implementation
├── Dataset_german_17_node.dat      # German network dataset
└── Dataset_european_19_node.dat    # European network dataset
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd qot-predictions

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run analysis
python qot_predictions_optical_networks.py
```

## 🏃♂️ Usage

### Basic Execution
The script performs:
1. **Data Loading**: Reads optical network datasets
2. **Feature Extraction**: Processes path and interference characteristics
3. **Statistical Analysis**: Computes dataset statistics
4. **Feature Matrix Creation**: Prepares data for ML models

### Expected Output
```
Length of Spans list: 5000
Length of Channels per link list: 5000
Length of SNR list: 5000

****************************************
Number of Spans: mean = 4.2, std = 1.8
Lightpath Length: mean = 850.5, std = 320.1
Max Channels per Link: mean = 12.3, std = 4.2
SNR: mean = 15.7, std = 3.4

Feature matrix shape: (5000, 8)
Target vector shape: (5000,)
```

## 📊 Dataset Format

**Input Format**: Semicolon-separated values
```
span_lengths;channel_counts;snr_value
80,120,90;10,15,12;14.5
```

**Feature Extraction**: 8 features per lightpath
1. Number of fiber spans along the path
2. Total lightpath length (km)
3. Longest fiber span length (km)
4. Maximum number of channels per link
5. Minimum number of channels per link
6. Mean number of channels per link
7. Number of links along the path
8. Total number of channels along the path

## 🎯 Project Objectives

### Core Tasks
1. **Implement Quantile Regression**: Use GBR with quantile loss for SNR prediction
2. **Implement Classification**: Use GBC for direct MF prediction
3. **Performance Comparison**: Analyze MF over/under-estimations between approaches
4. **Hyperparameter Optimization**: Grid search with cross-validation
5. **Statistical Analysis**: Comprehensive feature and performance analysis

### Evaluation Metrics
- **Regression**: R², MSE, Mean Pinball Loss
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Domain-specific**: MF over/under-estimation rates

## 🔧 Configuration

### Model Parameters
```python
# Quantile Regression
alpha = 0.1  # Quantile parameter (0.0-1.0)

# Hyperparameter Grid
param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'min_samples_split': [5, 10, 20]
}
```

## 📈 Extensions

The base implementation supports extension with:
- **Transfer Learning**: Domain adaptation between networks
- **Visualization**: Feature importance and prediction analysis
- **Advanced Models**: Ensemble methods and neural networks
- **Real-time Prediction**: Network planning integration

## 📚 Dependencies

- **numpy**: Numerical computing and array operations
- **matplotlib**: Data visualization and plotting
- **scikit-learn**: Machine learning algorithms and evaluation metrics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open Pull Request

## 📄 License

This project is part of the NDAL (Network Data Analytics Lab) coursework from Politecnico di Milano.

## 🔗 References

- [Scikit-learn Quantile Regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
- [IEEE Paper on QoT Prediction](https://ieeexplore.ieee.org/document/9355394)