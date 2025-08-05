# QoT Predictions - NDAL Project Group L

A machine learning project for Quality of Transmission (QoT) predictions in optical networks using Gradient Boosting algorithms.

## 📋 Project Description

This project implements machine learning models to predict Signal-to-Noise Ratio (SNR) values in optical networks based on lightpath characteristics. The analysis includes:

- Dataset preprocessing and feature extraction
- Statistical analysis of network parameters
- Machine learning model training and evaluation
- Performance metrics computation

## 🚀 Features

- **Data Processing**: Reads and processes optical network datasets
- **Feature Extraction**: Extracts 8 key features from lightpath data:
  - Number of fiber spans along the path
  - Total lightpath length
  - Longest fiber span length
  - Maximum/minimum/mean number of channels per link
  - Number of links along the path
  - Total number of channels along the path
- **Statistical Analysis**: Computes comprehensive statistics for all features
- **Machine Learning Ready**: Prepared for Gradient Boosting Regressor/Classifier models

## 📁 Project Structure

```
qot-predictions/
├── README.md
├── requirements.txt
├── ndal_project_groupl.py
├── Dataset_german_17_node.dat    # (add your dataset)
└── Dataset_european_19_node.dat  # (add your dataset)
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd qot-predictions
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your dataset files**
   - Place `Dataset_german_17_node.dat` in the project root
   - Place `Dataset_european_19_node.dat` in the project root
   - Or update the file paths in the `load_datasets()` function

## 🏃‍♂️ Running the Project

### Basic Usage

```bash
python ndal_project_groupl.py
```

### Expected Output

The script will:
1. Load and process the dataset
2. Display dataset statistics
3. Extract features and show matrix dimensions
4. Print summary statistics for all features

### Sample Output

```
Length of Spans list: 1000
Length of Channels per link list: 1000
Length of SNR list: 1000

****************************************
Number of Spans: mean = 4.2, std = 1.8
Lightpath Length: mean = 850.5, std = 320.1
Max Channels per Link: mean = 12.3, std = 4.2
SNR: mean = 15.7, std = 3.4

Feature matrix shape: (1000, 8)
Target vector shape: (1000,)
```

## 📊 Dataset Format

The expected dataset format is semicolon-separated with the following structure:
```
span1,span2,span3;channels1,channels2,channels3;snr_value
```

Example:
```
80,120,90;10,15,12;14.5
```

## 🔧 Configuration

### Updating Dataset Paths

Edit the `load_datasets()` function in `ndal_project_groupl.py`:

```python
def load_datasets():
    # Update these paths to your actual data file locations
    datafile_german = "path/to/your/Dataset_german_17_node.dat"
    datafile_european = "path/to/your/Dataset_european_19_node.dat"
    # ... rest of function
```

## 📈 Extending the Project

This base implementation can be extended with:

- **Machine Learning Models**: Add Gradient Boosting Regressor/Classifier training
- **Visualization**: Add matplotlib plots for data analysis
- **Model Evaluation**: Implement cross-validation and performance metrics
- **Transfer Learning**: Add domain adaptation capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is part of the NDAL (Network Data Analytics Lab) coursework.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all packages are installed via `pip install -r requirements.txt`
2. **File Not Found**: Ensure dataset files are in the correct location
3. **Permission Errors**: Use `pip install --user -r requirements.txt` if needed

### Getting Help

If you encounter issues:
1. Check that Python 3.8+ is installed
2. Verify all dependencies are installed correctly
3. Ensure dataset files are in the expected format
4. Check file paths in the `load_datasets()` function

## 📚 Dependencies

- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning algorithms and tools

For specific versions, see `requirements.txt`.