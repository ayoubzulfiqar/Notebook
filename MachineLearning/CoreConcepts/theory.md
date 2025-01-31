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

## Supervised Learning Algorithms

[Code](MachineLearning\CoreConcepts\supervised_learning.ipynb)

### **Supervised Learning Algorithms**

Supervised learning is a type of machine learning where the model is trained on labeled data, meaning the input data comes with corresponding output labels. The goal is to learn a mapping from inputs to outputs based on this data.

Supervised learning can be divided into two primary types: **Regression** (predicting continuous outcomes) and **Classification** (predicting discrete categories). Let's explore the key algorithms used in each category.

---

### **Regression Algorithms**

#### **1. Linear Regression**

**Concepts:**
- Linear regression models the relationship between a dependent variable $ y $ and one or more independent variables $ x $ by fitting a linear equation to the observed data.
- The general form of the linear regression model is:

$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
$

Where:
- $ \beta_0 $ is the intercept.
- $ \beta_1, \beta_2, \dots, \beta_n $ are the coefficients (weights) of the features.
- $ \epsilon $ is the error term.

**Assumptions:**
- Linearity: The relationship between the independent and dependent variables is linear.
- Independence: Observations are independent.
- Homoscedasticity: Constant variance of errors.
- Normality: Errors are normally distributed.

**Implementation:**
The parameters $ \beta_0, \beta_1, \dots $ are estimated using the **Ordinary Least Squares (OLS)** method, which minimizes the sum of squared residuals (differences between observed and predicted values):

$
\text{RSS} = \sum_{i=1}^n (y_i - \hat{y_i})^2
$

**Regularization:**
- **L1 Regularization (Lasso)**: Encourages sparsity by adding the absolute value of the coefficients to the cost function. The objective is to minimize:

$
\text{Cost} = \sum_{i=1}^n (y_i - \hat{y_i})^2 + \lambda \sum_{j=1}^p |\beta_j|
$

- **L2 Regularization (Ridge)**: Adds the squared value of coefficients to the cost function to prevent overfitting:

$
\text{Cost} = \sum_{i=1}^n (y_i - \hat{y_i})^2 + \lambda \sum_{j=1}^p \beta_j^2
$

#### **2. Polynomial Regression**

**Concept:**
- Polynomial regression is an extension of linear regression that allows for modeling non-linear relationships by adding polynomial terms of the input features. The equation becomes:

$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_1^2 + \dots + \beta_n x_1^n + \epsilon
$

Where $ x_1^2, x_1^3, \dots $ are higher-degree terms that capture non-linearity.

**Purpose:**
- This method helps when the data exhibits a non-linear relationship, but you still want to apply linear regression.

#### **3. Ridge and Lasso Regression**

- **Ridge Regression** (L2 Regularization): Helps prevent overfitting by penalizing large coefficients. The regularization term adds a penalty to the sum of squared coefficients.
- **Lasso Regression** (L1 Regularization): Similar to ridge, but it can result in sparse models (some coefficients become zero). Lasso is useful for feature selection.

The equations for Ridge and Lasso are as mentioned above.

---

### **Classification Algorithms**

#### **1. Logistic Regression**

**Concept:**
- Logistic regression is used for binary classification tasks, where the goal is to predict the probability of one class. The logistic function (sigmoid) maps the linear combination of input features to a probability:

$
p(y = 1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}
$

For multi-class classification, **one-vs-rest (OvR)** is used, where a separate binary classifier is trained for each class.

**Mathematical Representation:**
- Binary case: $ p = \frac{1}{1 + e^{-z}} $ where $ z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n $.

#### **2. Support Vector Machines (SVM)**

**Concept:**
- SVM is a powerful classifier that tries to find the optimal hyperplane that best separates the classes. The goal is to maximize the margin between the support vectors (data points closest to the hyperplane).

**Linear SVM:**
- The linear SVM tries to find a linear decision boundary:

$
w \cdot x + b = 0
$

Where $ w $ is the weight vector and $ b $ is the bias term.

**Kernel SVM:**
- For non-linearly separable data, the kernel trick is used to map data to a higher-dimensional space where it can be linearly separated.
- Common kernels include:
  - **Linear kernel**: $ k(x, x') = x \cdot x' $
  - **Polynomial kernel**: $ k(x, x') = (x \cdot x' + 1)^d $
  - **Radial Basis Function (RBF) kernel**: $ k(x, x') = e^{-\gamma ||x - x'||^2} $

#### **3. k-Nearest Neighbors (k-NN)**

**Concept:**
- k-NN is a simple classification algorithm that assigns the class label based on the majority class among the $ k $ nearest neighbors. The distance between points is calculated using metrics such as Euclidean or Manhattan distance.

**Mathematical Representation:**
- Distance metric (Euclidean):

$
d(x, x') = \sqrt{\sum_{i=1}^n (x_i - x'_i)^2}
$

**Parameter Tuning:**
- The key parameter is $ k $ (number of neighbors). A smaller $ k $ makes the algorithm sensitive to noise, while a larger $ k $ makes it smoother.

#### **4. Naïve Bayes**

**Concept:**
- Naïve Bayes is based on Bayes' Theorem and assumes that features are conditionally independent given the class. It's widely used in text classification tasks like spam detection.

Bayes' theorem:

$
P(C|X) = \frac{P(X|C)P(C)}{P(X)}
$

Where:
- $ P(C|X) $ is the posterior probability of class $ C $ given features $ X $.
- $ P(X|C) $ is the likelihood of features $ X $ given class $ C $.
- $ P(C) $ is the prior probability of class $ C $.
- $ P(X) $ is the evidence (normalizing constant).

#### **5. Decision Trees**

**Concept:**
- Decision trees split the data based on feature values to classify instances. They recursively partition the data and select the best feature to split on, based on certain criteria like **entropy** or **Gini index**.

**Mathematical Measures:**
- **Entropy** (measure of impurity or disorder):

$
H(D) = - \sum_{i=1}^{k} p_i \log_2 p_i
$

- **Gini Index** (another measure of impurity):

$
Gini(D) = 1 - \sum_{i=1}^{k} p_i^2
$

#### **6. Random Forest**

**Concept:**
- Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. Each tree is trained on a random subset of features and data (bootstrap sampling).

**Mathematical Representation:**
- Random forests aggregate the predictions of many decision trees, often using **majority voting** (for classification) or **averaging** (for regression).

---

### **Summary of Supervised Learning Algorithms**

#### **Regression Algorithms:**
- **Linear Regression**: Simple model, assumes linear relationships.
- **Polynomial Regression**: Extends linear regression for non-linear data.
- **Ridge and Lasso Regression**: Regularized linear models to prevent over-fitting.

#### **Classification Algorithms:**
- **Logistic Regression**: Binary and multi-class classification.
- **Support Vector Machines (SVM)**: Linear and kernel-based classifiers.
- **k-Nearest Neighbors (k-NN)**: Instance-based learning using distance metrics.
- **Naïve Bayes**: Probabilistic classifier, assumes feature independence.
- **Decision Trees**: Hierarchical classification based on feature splits.
- **Random Forest**: Ensemble of decision trees, improves robustness and accuracy.
