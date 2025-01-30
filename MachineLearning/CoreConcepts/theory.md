# Machine Learning Core Concept

## What is Machine Learning?
Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables systems to learn from data and make decisions without being explicitly programmed. ML algorithms improve their performance as they process more data, identifying patterns and making predictions or decisions based on input data.

Machine Learning is used in various applications, including recommendation systems, fraud detection, natural language processing (NLP), image recognition, and autonomous systems.

---

## Types of Machine Learning
Machine Learning is broadly categorized into three main types:

### 1. Supervised Learning
Supervised Learning involves training a model on a labeled dataset, meaning that each input data point has a corresponding correct output. The algorithm learns to map inputs to the correct outputs based on past examples.

#### Characteristics:
- Uses labeled data.
- The model learns from historical data.
- Can be used for classification and regression tasks.

#### Examples:
- Spam email detection (Classification)
- House price prediction (Regression)
- Handwritten digit recognition (Classification)

#### Common Algorithms:
- Linear Regression
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- Neural Networks

---

### 2. Unsupervised Learning
Unsupervised Learning involves training a model on data without labeled outputs. The algorithm tries to uncover hidden structures and patterns within the data.

#### Characteristics:
- No labeled data is required.
- Finds hidden patterns or structures in data.
- Used for clustering and dimensionality reduction.

#### Examples:
- Customer segmentation for marketing
- Anomaly detection in network security
- Topic modeling in NLP

#### Common Algorithms:
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- AutoEncoders

---

### 3. Reinforcement Learning
Reinforcement Learning (RL) is a type of learning where an agent interacts with an environment and learns by receiving rewards or penalties for its actions. The goal is to develop a strategy that maximizes cumulative rewards.

#### Characteristics:
- Involves an agent, an environment, actions, rewards, and a policy.
- The agent learns through trial and error.
- Commonly used in robotics and game-playing AI.

#### Examples:
- AlphaGo (playing board games)
- Self-driving cars
- Automated trading systems

#### Common Algorithms:
- Q-Learning
- Deep Q Networks (DQN)
- Policy Gradient Methods
- Proximal Policy Optimization (PPO)

---

## Key Terminology

### Features
Features are the measurable properties or characteristics of the data used as input for a machine learning model. They help the model make predictions or classifications.

**Example:** In a house price prediction model, features could include square footage, number of bedrooms, and location.

### Labels
Labels are the output or target variables in a supervised learning model. The model learns to predict labels based on the given features.

**Example:** In spam email detection, the label would be "spam" or "not spam."

### Training
Training refers to the process of teaching a machine learning model using a dataset. The model learns patterns from the data and optimizes its parameters to improve accuracy.

### Testing
Testing involves evaluating the trained model on unseen data to measure its performance. The test dataset helps determine how well the model generalizes to new examples.

### Validation
Validation is the process of fine-tuning the model by using a validation dataset. It helps in adjusting hyper-parameters and preventing over-fitting.

### Over-fitting
Over-fitting occurs when a model learns the training data too well, including noise and outliers, leading to poor generalization to new data.

**Example:** A model that memorizes training data but fails on test data.

### Under-fitting
Under-fitting happens when a model is too simple and fails to capture the underlying patterns in the data, resulting in poor performance on both training and test data.

**Example:** A linear model trying to fit highly non-linear data.

---

## Data Preprocessing in Machine Learning

[Code](MachineLearning\CoreConcepts\data_processing.ipynb)

## Data Cleaning
Data cleaning is the process of handling missing values, detecting and removing outliers, and correcting inconsistencies in data to improve the quality of input for machine learning models.

### Handling Missing Values
- **Removal:** Delete rows or columns with missing values (if the missing data is minimal).
- **Imputation:** Fill missing values using mean, median, mode, or predictive modeling (e.g., k-NN imputation).
- **Forward/Backward Filling:** Use previous or next values to fill gaps in time-series data.

### Outlier Detection and Removal
- **Statistical Methods:** Z-score, IQR (Inter-quartile Range) method.
- **Machine Learning-Based Methods:** Isolation Forest, DBSCAN clustering.
- **Manual Inspection:** Removing values that are domain-specific anomalies.

---

## Feature Scaling
Feature scaling standardizes or normalizes numerical features to bring them into a common range, improving model convergence and performance.

### Standardization
- Transforms features to have a mean of 0 and a standard deviation of 1.
- Formula: $ x' = \frac{x - \mu}{\sigma} $
- Used in algorithms like SVM, k-NN, PCA.

### Normalization (Min-Max Scaling)
- Scales values between a fixed range (usually 0 to 1).
- Formula: $ x' = \frac{x - x_{min}}{x_{max} - x_{min}} $
- Suitable for neural networks and distance-based algorithms.

---

## Encoding
Encoding is used to convert categorical data into numerical form for machine learning models.

### One-Hot Encoding
- Creates binary columns for each category.
- Useful for nominal categorical variables.

### Label Encoding
- Assigns integer values to categories.
- Can introduce ordinal relationships where none exist.

### Target Encoding
- Replaces categories with the mean of the target variable.
- Suitable for high-cardinality categorical features.

---

## Feature Selection
Feature selection removes irrelevant or redundant features to improve model performance and reduce over-fitting.

### Removing Irrelevant/Redundant Features
- **Variance Threshold:** Removes features with low variance.
- **Correlation Analysis:** Eliminates highly correlated features.

### Recursive Feature Elimination (RFE)
- Iteratively removes the least important features based on a model’s importance ranking.
- Commonly used with decision trees, SVM, and regression models.

---

## Dimensionality Reduction
Dimensionality reduction reduces the number of input features while retaining essential information.

### Principal Component Analysis (PCA)
- Projects data onto a lower-dimensional space using eigenvectors.
- Reduces redundancy while preserving variance.

### t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Non-linear technique for visualizing high-dimensional data.
- Retains local structure of data while reducing dimensions.

### Linear Discriminant Analysis (LDA)
- Used for supervised dimensionality reduction.
- Maximizes class separability while reducing dimensions.

---

## **Algorithms for Data Processing**
 


### **1. Data Cleaning**  
- Handling Missing Values:  
  - Mean/Median/Mode Imputation  
  - K-Nearest Neighbors (KNN) Imputation  
  - Interpolation  
- Outlier Detection and Removal:  
  - Z-score Method  
  - IQR (Inter-quartile Range) Method  
  - Isolation Forest  
  - Local Outlier Factor (LOF)  

### **2. Feature Scaling**  
- Standardization (Z-score Normalization)  
- Min-Max Scaling (Normalization)  
- Robust Scaling (for handling outliers)  
- Log Transformation  
- Power Transformation (Box-Cox, Yeo-Johnson)  

### **3. Encoding Categorical Data**  
- One-Hot Encoding  
- Label Encoding  
- Target Encoding  
- Frequency Encoding  
- Binary Encoding  
- Ordinal Encoding  

### **4. Feature Selection**  
- Filter Methods:  
  - Mutual Information  
  - Chi-Square Test  
  - ANOVA Test  
- Wrapper Methods:  
  - Recursive Feature Elimination (RFE)  
  - Forward/Backward Feature Selection  
- Embedded Methods:  
  - LASSO (L1 Regularization)  
  - Decision Tree Feature Importance  
  - SHAP (SHapley Additive exPlanations)  

### **5. Dimensionality Reduction**  
- Principal Component Analysis (PCA)  
- Linear Discriminant Analysis (LDA)  
- t-Distributed Stochastic Neighbor Embedding (t-SNE)  
- AutoEncoders  
- Independent Component Analysis (ICA)  

### **6. Data Transformation & Augmentation**  
- Polynomial Features  
- Discretization (Binning)  
- Data Augmentation (for image and text data)  
  - Image: Flipping, Rotation, Scaling  
  - Text: Synonym Replacement, Back Translation  

---