# **Core Machine Learning Concepts** 

---

### **1. Basics of Machine Learning**
- **What is Machine Learning?**: Understanding the types (Supervised, Unsupervised, Reinforcement Learning).  
- **Key Terminology**: Features, Labels, Training, Testing, Validation, Over-fitting, Under-fitting.  
- **Applications of ML**: Real-world use cases like recommendation systems, fraud detection, NLP, etc.

---

### **2. Data Preprocessing**
- **Data Cleaning**: Handling missing values, outlier detection and removal.  
- **Feature Scaling**: Standardization, Normalization (Min-Max scaling).  
- **Encoding**: One-hot encoding, Label encoding, Target encoding.  
- **Feature Selection**: Removing irrelevant/redundant features, Recursive Feature Elimination (RFE).  
- **Dimensionality Reduction**: PCA, t-SNE, LDA.  

---

### **3. Supervised Learning Algorithms**
#### Regression:
- **Linear Regression**: Concepts, assumptions, implementation, regularization (L1, L2).  
- **Polynomial Regression**: Extending linear regression for non-linear data.  
- **Ridge and Lasso Regression**: Regularization techniques to handle over-fitting.  

#### Classification:
- **Logistic Regression**: Binary classification, multi-class classification using one-vs-rest (OvR).  
- **Support Vector Machines (SVM)**: Linear and kernel-based SVMs.  
- **k-Nearest Neighbors (k-NN)**: Distance metrics, parameter tuning.  
- **Naïve Bayes**: Applications in text classification, spam detection.  
- **Decision Trees**: Entropy, Gini index, over-fitting prevention (pruning).  
- **Random Forests**: Ensemble learning using multiple decision trees.  

---

### **4. Unsupervised Learning Algorithms**
- **Clustering**:  
  - **K-means Clustering**: Elbow method for optimal  (k).  
  - **Hierarchical Clustering**: Agglomerative and divisive methods.  
  - **DBSCAN**: Density-based clustering.  
- **Dimensionality Reduction**:  
  - **PCA**: Reducing features while preserving variance.  
  - **t-SNE**: Visualization of high-dimensional data.  
- **Association Rules**: Market basket analysis, Apriori algorithm.  

---

### **5. Evaluation Metrics**
#### Regression Metrics:
- Mean Absolute Error (MAE).  
- Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).  
- R-squared ($ R^2 $) and Adjusted $ R^2 $.  

#### Classification Metrics:
- Accuracy, Precision, Recall, F1-Score.  
- Confusion Matrix.  
- ROC-AUC Curve, PR-AUC Curve.  

#### Clustering Metrics:
- Silhouette Score.  
- Inertia (for K-means).  

---

### **6. Model Selection and Validation**
- **Train-Test Split**: Partitioning data into training and testing sets.  
- **Cross-Validation**: k-Fold, Stratified k-Fold, Leave-One-Out (LOOCV).  
- **Bias-Variance Tradeoff**: Understanding model generalization.  
- **Grid Search and Random Search**: Hyperparameter tuning.  

---

### **7. Feature Engineering**
- **Interaction Terms**: Creating new features from existing ones.  
- **Polynomial Features**: Enhancing feature space for non-linear models.  
- **Binning**: Discretizing continuous variables.  
- **Feature Importance**: Using algorithms to rank feature contributions.  

---

### **8. Ensemble Methods**
- **Bagging**: Bootstrap aggregating (e.g., Random Forest).  
- **Boosting**:  
  - AdaBoost.  
  - Gradient Boosting (GBM).  
  - XGBoost, LightGBM, CatBoost.  
- **Stacking**: Combining multiple models for better performance.  

---

### **9. Regularization Techniques**
- **L1 Regularization (Lasso)**: Feature selection and sparsity.  
- **L2 Regularization (Ridge)**: Penalizing large coefficients.  
- **Dropout**: Preventing over-fitting in neural networks.  

---

### **10. Time Series Analysis**
- **Stationarity**: Dickey-Fuller test.  
- **AutoRegressive Models**: ARIMA, SARIMA.  
- **Seasonality and Trends**: Decomposition techniques.  


---

<!-- 
### **11. Advanced Topics (Optional for Now)**
- **Reinforcement Learning**: Q-learning, Deep Q-Networks.  
- **Deep Learning**: Introduction to neural networks, back-propagation.  
- **NLP Concepts**: Tokenization, word embeddings, transformers.   -->