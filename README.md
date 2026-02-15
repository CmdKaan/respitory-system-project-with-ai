# respitory-system-project-with-ai
The project utilizes a hybrid AI approach to process different data types:
CNN (Convolutional Neural Networks): Analyzes respiratory sounds (wheezing, crackles) converted into spectrograms.
XGBoost: Processes numerical data (SpO2, CO2 and airflow) to calculate a "health risk score".
LSTM (Long Short-Term Memory): Performs trend analysis on historical data to predict capacity changes over time.Optimization: Focuses on creating lightweight models for efficient execution on edge devices (TinyML).
Week 1: Signal Processing & Feature Extraction
Objective: Implement the initial acoustic processing pipeline.
Achievement: Developed a Python-based class for MFCC (Mel-Frequency Cepstral Coefficient) extraction.
Details: MFCC is utilized as the primary feature set for the CNN model, as research indicates it can achieve high accuracy (up to 99%) in classifying respiratory diseases.
Status: Successfully extracting acoustic signatures from raw audio data for model training.
