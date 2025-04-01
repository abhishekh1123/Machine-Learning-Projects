# Diabetes Prediction using SVM

## Overview
This project implements a **Diabetes Prediction System** using **Support Vector Machine (SVM)** as the classification model. The dataset used is the **PIMA Diabetes Dataset**, which contains medical records of patients along with their diabetes outcomes.

## Dataset
The dataset consists of **768** rows and **9** columns, including features like **Glucose, Blood Pressure, BMI, Age, Insulin levels**, and a target column **Outcome** (1 = Diabetic, 0 = Non-Diabetic).

## Installation & Dependencies
To run this project, install the required Python libraries:
```bash
pip install numpy pandas scikit-learn
```

## Steps Involved

### 1. Importing Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### 2. Load the Dataset
```python
diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.head())
```

### 3. Data Preprocessing
- Separate features and target variable:
```python
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```
- Standardize the data:
```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### 4. Splitting Data
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### 5. Train the Model
```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

### 6. Evaluate the Model
```python
training_accuracy = accuracy_score(classifier.predict(X_train), Y_train)
test_accuracy = accuracy_score(classifier.predict(X_test), Y_test)
print(f'Training Accuracy: {training_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
```

### 7. Making a Prediction
```python
input_data = (8,176,90,34,300,33.7,0.467,58)
input_data_np = np.asarray(input_data).reshape(1, -1)
input_std = scaler.transform(input_data_np)
prediction = classifier.predict(input_std)
if prediction[0] == 0:
    print('The person is non-diabetic.')
else:
    print('The person is diabetic.')
```

## Results
- **Training Accuracy:** 78.66%
- **Test Accuracy:** 77.27%

## Future Enhancements
- Experiment with other classification algorithms (Random Forest, Neural Networks, etc.)
- Improve feature selection and data cleaning.
- Deploy the model as a web application.

## Conclusion
This project demonstrates a **machine learning-based approach** to predicting diabetes using **SVM classification**. The model achieves a **good accuracy** and can be further enhanced with better feature engineering and hyperparameter tuning.

---
Developed by: [Your Name]

