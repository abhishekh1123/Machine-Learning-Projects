
```markdown
# Sonar Rock vs Mine Prediction using Logistic Regression

## Overview
This project implements a **Sonar Signal Classification System** using **Logistic Regression** as the classification model. The dataset contains sonar signals bounced off either **Rocks (R)** or **Metal Cylinders (Mines)**.

## Dataset
The dataset consists of:
- **208 samples** (111 Mines, 97 Rocks)
- **60 numerical features** representing energy in different frequency bands
- **1 target column** (R = Rock, M = Mine)

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 2. Load the Dataset
```python
sonar_data = pd.read_csv('Sonar data.csv', header=None)
print(sonar_data.head())
```

### 3. Data Preprocessing
- Separate features and target variable:
```python
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
```

### 4. Splitting Data
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
```

### 5. Train the Model
```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

### 6. Evaluate the Model
```python
training_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)
print(f'Training Accuracy: {training_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
```

### 7. Making a Prediction
```python
input_data = (0.0374,0.0586,0.0628,0.0534,0.0255,0.1422,0.2072,0.2734,0.3070,0.2597,
              0.3483,0.3999,0.4574,0.5950,0.7924,0.8272,0.8087,0.8977,0.9828,0.8982,
              0.8890,0.9367,0.9122,0.7936,0.6718,0.6318,0.4865,0.3388,0.4832,0.3822,
              0.3075,0.1267,0.0743,0.1510,0.1906,0.1817,0.1709,0.0946,0.2829,0.3006,
              0.1602,0.1483,0.2875,0.2047,0.1064,0.1395,0.1065,0.0527,0.0395,0.0183,
              0.0353,0.0118,0.0063,0.0237,0.0032,0.0087,0.0124,0.0113,0.0098,0.0126)
              
input_data_np = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_np)

if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a Mine')
```

## Results
- **Training Accuracy:** 83.42%
- **Test Accuracy:** 76.19%

## Future Enhancements
- Experiment with other classification algorithms (SVM, Random Forest, Neural Networks)
- Implement feature selection to improve accuracy
- Add data visualization for better insights
- Deploy the model as a web application

## Conclusion
This project demonstrates a **machine learning-based approach** to classifying sonar signals using **Logistic Regression**. The model achieves **good accuracy** and can be enhanced with feature engineering and hyperparameter tuning for naval mine detection applications.
```

