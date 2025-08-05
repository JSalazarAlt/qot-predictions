#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify all imports work correctly"""

try:
    import numpy as np
    print("[OK] numpy imported successfully")
    
    import matplotlib.pyplot as plt
    print("[OK] matplotlib.pyplot imported successfully")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    print("[OK] sklearn modules imported successfully")
    
    # Test basic functionality
    arr = np.array([1, 2, 3, 4, 5])
    print(f"[OK] numpy array created: {arr}")
    
    scaler = StandardScaler()
    print("[OK] StandardScaler instantiated")
    
    print("\n[SUCCESS] All imports and basic functionality working!")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
except Exception as e:
    print(f"[ERROR] Other error: {e}")