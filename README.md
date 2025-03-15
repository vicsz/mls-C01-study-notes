# mls-C01-study-notes

# **Amazon SageMaker - Exam Cheat Sheet**

## **General Concepts**
- **SageMaker is a fully managed ML service** that helps with **building, training, and deploying models**.
- **Supports built-in algorithms, custom models, and frameworks like TensorFlow, PyTorch, and MXNet.**
- **For managed infrastructure and reduced ops, use**:
  - **SageMaker Autopilot** → Auto ML
  - **JumpStart** → Pre-trained models
  - **Pipelines** → ML workflow automation

---

## **SageMaker Core Components**
1. **SageMaker Studio** → Integrated IDE for ML development.
2. **SageMaker Notebooks** → Jupyter-based notebooks with managed infrastructure.
3. **SageMaker Training** → Distributed model training.
4. **SageMaker Inference** → Model hosting for real-time & batch inference.
5. **SageMaker Feature Store** → Centralized repository for storing and sharing features.
6. **SageMaker Model Monitor** → Detects data drift and monitors model performance.
7. **SageMaker Debugger** → Debugs and profiles ML models during training.
8. **SageMaker Pipelines** → Automates ML workflows.
9. **SageMaker Autopilot** → Automates model training and tuning.
10. **SageMaker JumpStart** → Pre-trained models and solutions.
11. **SageMaker Data Wrangler** → Prepares and transforms ML data.

---

## **SageMaker Training**
### **Instance Types**
- **GPU Instances**: Use `ml.p3`, `ml.p4`, `ml.g5` for deep learning.
- **CPU Instances**: Use `ml.m5`, `ml.c5` for traditional ML.
- **Multi-GPU Training**: Use **Distributed Training** feature.
- **Spot Instances**: Saves **up to 90%** on training costs but jobs can be interrupted.

### **Training Storage Options**
- **Amazon S3** → Default option, best for large datasets.
- **Amazon FSx for Lustre** → Best for high-performance distributed training.
- **Amazon EFS** → Good for sharing data across training instances.

### **Hyperparameter Optimization (HPO)**
- Uses **Bayesian optimization** to find the best hyperparameters.
- **Rule of Thumb**: Use **SageMaker Automatic Model Tuning** when optimizing hyperparameters.

---

## **SageMaker Deployment & Inference**
### **Inference Types**
- **Real-time Inference** → Deploys an endpoint with auto-scaling.
- **Batch Transform** → Processes large datasets at once, useful for stored data.
- **Asynchronous Inference** → Queues requests when response time is not critical.
- **Serverless Inference** → Cost-efficient for intermittent workloads.

### **Instance Selection for Deployment**
- **CPU for light workloads** → `ml.m5`, `ml.c5`
- **GPU for deep learning** → `ml.g4dn`, `ml.inf1` for inference

### **Model Deployment Strategies**
- **Single Model Endpoint** → One model per endpoint.
- **Multi-Model Endpoint (MME)** → Host multiple models per endpoint to save costs.
- **Shadow Deployment** → Test a new model in parallel with the existing one.
- **Blue/Green Deployment** → Gradually switch between old and new models.

---

## **SageMaker Data Processing**
- **SageMaker Data Wrangler** → Simplifies data cleaning and transformation.
- **SageMaker Processing Jobs** → Runs preprocessing jobs (ETL, feature engineering).
- **SageMaker Feature Store** → Stores, retrieves, and shares ML features.
- **SageMaker Ground Truth** → Manages human-labeling tasks for datasets.

---

## **SageMaker Security & Governance**
- **IAM Policies** → Control access to SageMaker resources.
- **VPC Configurations** → SageMaker can run within a private VPC for security.
- **Encryption**:
  - **Data at Rest** → Encrypt using Amazon S3/KMS.
  - **Data in Transit** → Use TLS encryption.

---

## **SageMaker Cost Optimization**
- **Use Spot Training** → Saves up to **90%** costs for training.
- **Use Multi-Model Endpoints** → Saves inference costs by hosting multiple models per endpoint.
- **Use Serverless Inference** → Best for low-traffic workloads.

---

## **SageMaker Monitoring & Debugging**
- **SageMaker Model Monitor**
  - Tracks **data drift, model performance degradation, bias detection**.
  - Supports **baseline generation** for comparisons.
- **SageMaker Debugger**
  - Monitors training jobs for inefficiencies like **overfitting, underfitting, NaN losses**.

---

## **Exam Rules of Thumb**
- **Batch Inference** → Use **Batch Transform** (not real-time inference).
- **Low-cost inference** → Use **Multi-Model Endpoints or Serverless Inference**.
- **Optimize training costs** → Use **Spot Instances**.
- **Avoid data drift** → Use **SageMaker Model Monitor**.
- **For feature storage** → Use **SageMaker Feature Store**.
- **Automate ML pipeline** → Use **SageMaker Pipelines**.
- **Want an easy way to train models?** → Use **SageMaker Autopilot**.
- **Need to track/debug training performance?** → Use **SageMaker Debugger**.
- **For labeling data at scale** → Use **SageMaker Ground Truth**.

# **Amazon SageMaker Algorithms - Exam Cheat Sheet**

## **Supervised Learning (Labeled Data)**

### **1. Linear Learner (Regression & Classification)**
- **Use Case**: Simple regression (predicting continuous values) or classification (categorical predictions).
- **Example**: Predicting house prices, fraud detection.
- **Rule of Thumb**: Use for simple structured data problems when deep learning is unnecessary.

### **2. XGBoost (Extreme Gradient Boosting)**
- **Use Case**: Highly optimized decision tree-based model for structured/tabular data.
- **Example**: Fraud detection, credit risk scoring, ranking models.
- **Rule of Thumb**: Go-to for most structured/tabular data tasks due to its efficiency and accuracy.

### **3. Seq2Seq (Sequence-to-Sequence)**
- **Use Case**: Natural Language Processing (NLP) for input-output sequence mapping.
- **Example**: Machine translation, chatbot responses.
- **Rule of Thumb**: Use for NLP tasks requiring input-output sequence pairs.

### **4. DeepAR (Time Series Forecasting)**
- **Use Case**: Forecasting numerical time series data with deep learning.
- **Example**: Sales forecasting, demand prediction.
- **Rule of Thumb**: Use when dealing with multiple time series and needing more accuracy than traditional models.

---

## **Unsupervised Learning (No Labels)**

### **5. K-Means Clustering**
- **Use Case**: Grouping similar data points together (clustering).
- **Example**: Customer segmentation, anomaly detection.
- **Rule of Thumb**: Use when you need to categorize unlabeled data into meaningful clusters.

### **6. Principal Component Analysis (PCA)**
- **Use Case**: Dimensionality reduction for large datasets.
- **Example**: Reducing image size while preserving key information, preprocessing for faster model training.
- **Rule of Thumb**: Use when working with high-dimensional data to remove redundant features.

---

## **Anomaly Detection**

### **7. Random Cut Forest (RCF)**
- **Use Case**: Detecting outliers or anomalies in data.
- **Example**: Fraud detection, network security monitoring.
- **Rule of Thumb**: Use when you need to find unusual patterns in a dataset.

---

## **Natural Language Processing (NLP)**

### **8. BlazingText**
- **Use Case**: Fast and efficient text classification and word embeddings.
- **Example**: Sentiment analysis, spam detection.
- **Rule of Thumb**: Use when processing large text datasets efficiently.

### **9. Object2Vec**
- **Use Case**: Embedding objects (text, users, products) into vector space for similarity analysis.
- **Example**: Product recommendations, personalized search.
- **Rule of Thumb**: Use when you need to map relationships between objects.

---

## **Computer Vision**

### **10. Semantic Segmentation**
- **Use Case**: Classifying each pixel in an image for object detection.
- **Example**: Self-driving car lane detection, medical image analysis.
- **Rule of Thumb**: Use for pixel-level image classification (not just object bounding boxes).

---

## **Additional Algorithms in These Categories**

### **Supervised Learning**
- **LightGBM** → Similar to XGBoost, but faster on large datasets with categorical features.
- **Support Vector Machines (SVM)** → Binary classification with clear margin separation.

### **Unsupervised Learning**
- **DBSCAN** → Density-based clustering for non-uniform data.
- **Hierarchical Clustering** → Tree-based clustering for data organization.

### **Anomaly Detection**
- **Isolation Forest** → Tree-based anomaly detection with fast performance.

### **NLP**
- **BERT** → Pre-trained deep learning model for contextual NLP tasks.
- **GPT** → Text generation, conversational AI.

### **Computer Vision**
- **YOLO** → Real-time object detection.
- **Faster R-CNN** → High-accuracy object detection.

### **Time Series Forecasting**
- **Prophet** → Forecasting with seasonal effects.
- **ARIMA** → Simple time series prediction with autoregressive models.

### **Reinforcement Learning**
- **Deep Q-Network (DQN)** → AI decision-making in dynamic environments.


