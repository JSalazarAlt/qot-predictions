#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDAL Project Group L - QoT Predictions
Converted from Jupyter notebook to Python script
"""

# LIBRARIES IMPORTATION

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import max_error, mean_squared_error, mean_pinball_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from itertools import chain
from pprint import pprint

# 1. READ DATASET

# 1A. 'Read Dataset' Function Definition
# The function read_dataset() reads a file passed in input and returns three lists with lightpaths characteristics:
# - Matrix of span lengths.
# - Matrix of number of channels per link.
# - Vector of Signal-to-Noise Ratios (SNRs).

def read_dataset(filename):
    """
    Read dataset from file and extract lightpath characteristics.
    
    Args:
        filename: Name of data file.
    
    Returns:
        span_length_matrix: List with spans lengths for all lightpaths
        channels_link_matrix: List with number of channels per link info
        SNR_vect: List of SNR values for each lightpath [dB]
    """

    span_length_matrix = []
    channels_link_matrix = []
    SNR_vect = []

    with open(filename) as data_file:

        next(data_file)

        for line in data_file:
            elements = line.split(';', 3)
            
            # Extract span lengths
            spans_list = elements[0].split(',')
            span_lengths_vect = [int(span) for span in spans_list]
            span_length_matrix.append(span_lengths_vect)
            
            # Extract channels per link
            channels_list = elements[1].split(',')
            channels_link_vect = [int(channel) for channel in channels_list]
            channels_link_matrix.append(channels_link_vect)
            
            # Extract SNR
            snr = float(elements[2])
            SNR_vect.append(snr)

    return span_length_matrix, channels_link_matrix, SNR_vect

# 1B. 'Read Dataset' Function Call and Statistics Computation

def load_datasets():
    """Load and combine datasets from German and European networks."""
    # Update these paths to your actual data file locations
    datafile_german = "Dataset_german_17_node.dat"
    datafile_european = "Dataset_european_19_node.dat"
    
    # Read datafiles
    spans_german, channels_link_german, snr_values_german = read_dataset(datafile_german)
    # spans_european, channels_link_european, snr_values_european = read_dataset(datafile_european)
    
    # For now, using only German dataset
    spans = spans_german
    channels_link = channels_link_german
    snr_values = snr_values_german
    
    return spans, channels_link, snr_values

def compute_statistics(spans, channels_link, snr_values):
    """Compute and print dataset statistics."""
    print(f'Length of Spans list: {len(spans)}')
    print(f'Length of Channels per link list: {len(channels_link)}')
    print(f'Length of SNR list: {len(snr_values)}')
    
    # Statistics from spans
    numspans = [len(span) for span in spans]
    lightpathlength = [sum(span) for span in spans]
    
    mean_numspans = round(np.mean(numspans), 2)
    std_numspans = round(np.std(numspans), 2)
    
    mean_lightpathlength = round(np.mean(lightpathlength), 2)
    std_lightpathlength = round(np.std(lightpathlength), 2)
    
    # Statistics from channels
    maxnuminterf = [max(channels) for channels in channels_link]
    mean_maxnuminterf = round(np.mean(maxnuminterf), 2)
    std_maxnuminterf = round(np.std(maxnuminterf), 2)
    
    # Statistics from SNR
    mean_snr = round(np.mean(snr_values), 2)
    std_snr = round(np.std(snr_values), 2)
    
    print('\n' + '*'*40)
    print(f'Number of Spans: mean = {mean_numspans}, std = {std_numspans}')
    print(f'Lightpath Length: mean = {mean_lightpathlength}, std = {std_lightpathlength}')
    print(f'Max Channels per Link: mean = {mean_maxnuminterf}, std = {std_maxnuminterf}')
    print(f'SNR: mean = {mean_snr}, std = {std_snr}')

# 2. FEATURES EXTRACTION

# 2A. 'Extract Features' Function Definition
# Extracts 8 features:
# 1. Number of fiber spans along the path
# 2. Total lightpath length
# 3. Longest fiber span length
# 4. Maximum number of channels per link
# 5. Minimum number of channels per link
# 6. Mean number of channels per link
# 7. Number of links along the path
# 8. Total number of channels along the path

def extract_features(span_matrix, channels_link_matrix):
    """
    Extract features from span and channel matrices.
    
    Args:
        span_matrix: Matrix of spans for each lightpath
        channels_link_matrix: Matrix of channels per link info
    
    Returns:
        numpy array with 8 features per lightpath
    """
    X_matrix = []
    
    for span_len_vect, channels_link_vect in zip(span_matrix, channels_link_matrix):
        feature_vect = []
        
        # Features from span vector
        feature_vect.append(len(span_len_vect))  # Number of spans
        feature_vect.append(sum(span_len_vect))  # Total length
        feature_vect.append(max(span_len_vect))  # Longest span
        
        # Features from channels vector
        feature_vect.append(max(channels_link_vect))  # Max channels per link
        feature_vect.append(min(channels_link_vect))  # Min channels per link
        feature_vect.append(np.mean(channels_link_vect))  # Mean channels per link
        feature_vect.append(len(channels_link_vect))  # Number of links
        feature_vect.append(sum(channels_link_vect))  # Total channels
        
        X_matrix.append(feature_vect)
    
    return np.array(X_matrix)


def main():
    """Main function to run the analysis."""
    try:
        # Load datasets
        spans, channels_link, snr_values = load_datasets()
        
        # Compute statistics
        compute_statistics(spans, channels_link, snr_values)
        
        # Extract features
        X = extract_features(spans, channels_link)
        y = np.array(snr_values)
        
        print(f'\nFeature matrix shape: {X.shape}')
        print(f'Target vector shape: {y.shape}')
        
        return X, y
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please update the file paths in load_datasets().")
        print(f"Details: {e}")
        return None, None


if __name__ == "__main__":
    X, y = main()