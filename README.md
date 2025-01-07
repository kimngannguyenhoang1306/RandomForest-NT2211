# **RandomForest-NT2211**

This project demonstrates a Machine Learning pipeline using Random Forest Classifier for structured data. It includes preprocessing, model training, evaluation, and artifact management.

## **Features**
- Automatic preprocessing for both numeric and categorical data.
- Cross-validation for performance estimation.
- Random Forest Classifier with hyperparameter optimization.
- Comprehensive logging and reporting.
- Support for large datasets with memory-efficient operations.

---

## **Requirements**

### **1. Install Python and Required Packages**
- Ensure Python 3.8+ is installed.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## **Usage**

### **1. Prepare the Data**
- Place your training data (`train.csv`) and testing data (`test.csv`) in the root directory (equal to the directory of main.py).

### **2. Run the Pipeline**
- Execute the `main.py` script to train and evaluate the model:
  ```bash
  python main.py
  ```
- The pipeline will:
  - Load and preprocess the data.
  - Train a Random Forest model.
  - Optimize hyperparameters.
  - Save the trained model and evaluation reports.

### **3. Output**
- Artifacts (trained model, scaler, label encoder) will be saved in the `outputs/` directory.
- Logs and evaluation reports will be stored with timestamped filenames.

---

## **File Structure**
```plaintext
RandomForest-NT2211/
├── train.csv            # Training data
├── test.csv             # Testing data
├── main.py              # Entry point for the pipeline
├── requirements.txt     # List of required Python libraries
├── outputs/             # Directory for output artifacts and logs
│   ├── model_*.joblib   # Saved model
│   ├── scaler_*.joblib  # Saved scaler
│   ├── logs/            # Logs and evaluation reports
├── README.md            # Project documentation
├── results              # Samples results
```

---

## **Key Components**
### **1. `main.py`**
Main entry point of the project. It initializes the pipeline, loads data, trains the model, and handles artifacts.

### **2. `requirements.txt`**
Contains the list of required Python libraries for the project:
```plaintext
numpy
pandas
scikit-learn
joblib
```

---
