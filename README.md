# AWS Certified Machine Learning - Specialty (MLS-C01) Exam Study Notes

# Domain 1: Data Engineering 
## **Task Statement 1.1: Create Data Repositories for ML**

### **1. Data Storage Options for ML**
- **Amazon S3** → Scalable object storage for large datasets.
- **Amazon Redshift** → Columnar data warehouse optimized for analytical queries.
- **Amazon RDS** → Managed relational database for structured data.
- **Amazon DynamoDB** → NoSQL key-value store for high-speed low-latency queries.
- **Amazon Timestream** → Optimized for time-series data (sensor data, logs).
- **Amazon FSx for Lustre** → High-performance file storage for ML workloads.
- **Amazon EFS** → Fully managed file system for shared access.
- **AWS Glue Data Catalog** → Centralized metadata store for ML datasets.

### **2. Choosing the Right Storage for ML**
| **Use Case**               | **Best Storage Option**        |
|---------------------------|--------------------------------|
| Large-scale object storage | **Amazon S3**                 |
| Fast analytical queries    | **Amazon Redshift**           |
| Structured relational data | **Amazon RDS**                |
| NoSQL key-value lookups    | **Amazon DynamoDB**           |
| Time-series data           | **Amazon Timestream**         |
| High-performance training  | **Amazon FSx for Lustre**     |
| Shared file storage        | **Amazon EFS**                |
| Metadata management        | **AWS Glue Data Catalog**     |

### **3. Data Lake Architecture**
- **Amazon S3** → Primary storage for raw, processed, and enriched ML data.
- **AWS Lake Formation** → Automates security, permissions, and access control for data lakes.
- **AWS Glue** → Extract, Transform, Load (ETL) service to prepare ML data.
- **Amazon Athena** → Query data in S3 using SQL (serverless).
- **Amazon Redshift Spectrum** → Query S3 data using Redshift.
- **Amazon EMR** → Process large-scale data using Spark, Hadoop.

### **4. Security & Access Control**
- **IAM Roles & Policies** → Restrict access to ML data repositories.
- **Amazon Macie** → Detects sensitive data in S3 (PII, credentials).
- **AWS Key Management Service (KMS)** → Encrypts ML data at rest.
- **VPC Endpoints** → Secure access to S3 without internet exposure.

### **5. Data Governance & Compliance**
- **AWS Lake Formation** → Simplifies access control.
- **AWS Glue Data Catalog** → Maintains metadata consistency.
- **Amazon Macie** → Identifies security risks in stored data.
- **AWS Audit Manager** → Helps with compliance tracking.

---

## **Exam Rules of Thumb**
- **For ML data storage → Use S3** (most scalable & cost-effective).
- **For fast analytical queries → Use Redshift**.
- **For structured relational data → Use RDS**.
- **For high-speed NoSQL lookups → Use DynamoDB**.
- **For time-series ML data → Use Timestream**.
- **For high-performance ML training → Use FSx for Lustre**.
- **For shared access file storage → Use EFS**.
- **For metadata management → Use AWS Glue Data Catalog**.
- **For securing ML data → Use Macie, KMS, IAM, and Lake Formation**.

## **Task Statement 1.1: Create Data Repositories for ML**

### **1. Data Storage Options for ML**
- **Amazon S3** → Scalable object storage for large datasets.
- **Amazon Redshift** → Columnar data warehouse optimized for analytical queries.
- **Amazon RDS** → Managed relational database for structured data.
- **Amazon DynamoDB** → NoSQL key-value store for high-speed low-latency queries.
- **Amazon Timestream** → Optimized for time-series data (sensor data, logs).
- **Amazon FSx for Lustre** → High-performance file storage for ML workloads.
- **Amazon EFS** → Fully managed file system for shared access.
- **AWS Glue Data Catalog** → Centralized metadata store for ML datasets.

### **2. Choosing the Right Storage for ML**
| **Use Case**               | **Best Storage Option**        |
|---------------------------|--------------------------------|
| Large-scale object storage | **Amazon S3**                 |
| Fast analytical queries    | **Amazon Redshift**           |
| Structured relational data | **Amazon RDS**                |
| NoSQL key-value lookups    | **Amazon DynamoDB**           |
| Time-series data           | **Amazon Timestream**         |
| High-performance training  | **Amazon FSx for Lustre**     |
| Shared file storage        | **Amazon EFS**                |
| Metadata management        | **AWS Glue Data Catalog**     |

### **3. Data Lake Architecture**
- **Amazon S3** → Primary storage for raw, processed, and enriched ML data.
- **AWS Lake Formation** → Automates security, permissions, and access control for data lakes.
- **AWS Glue** → Extract, Transform, Load (ETL) service to prepare ML data.
- **Amazon Athena** → Query data in S3 using SQL (serverless).
- **Amazon Redshift Spectrum** → Query S3 data using Redshift.
- **Amazon EMR** → Process large-scale data using Spark, Hadoop.

### **4. Security & Access Control**
- **IAM Roles & Policies** → Restrict access to ML data repositories.
- **Amazon Macie** → Detects sensitive data in S3 (PII, credentials).
- **AWS Key Management Service (KMS)** → Encrypts ML data at rest.
- **VPC Endpoints** → Secure access to S3 without internet exposure.

### **5. Data Governance & Compliance**
- **AWS Lake Formation** → Simplifies access control.
- **AWS Glue Data Catalog** → Maintains metadata consistency.
- **Amazon Macie** → Identifies security risks in stored data.
- **AWS Audit Manager** → Helps with compliance tracking.

---

## **Exam Rules of Thumb**
- **For ML data storage → Use S3** (most scalable & cost-effective).
- **For fast analytical queries → Use Redshift**.
- **For structured relational data → Use RDS**.
- **For high-speed NoSQL lookups → Use DynamoDB**.
- **For time-series ML data → Use Timestream**.
- **For high-performance ML training → Use FSx for Lustre**.
- **For shared access file storage → Use EFS**.
- **For metadata management → Use AWS Glue Data Catalog**.
- **For securing ML data → Use Macie, KMS, IAM, and Lake Formation**.

## **Task Statement 1.3: Identify and Implement a Data Transformation Solution**

### **1. Data Transformation Services**
- **AWS Glue** → Fully managed ETL (Extract, Transform, Load) service.
- **Amazon EMR** → Hadoop/Spark-based big data processing.
- **AWS Lambda** → Serverless transformations for real-time processing.
- **Amazon Kinesis Data Analytics** → SQL-based real-time stream processing.
- **Amazon Redshift Spectrum** → Query & transform S3 data using Redshift.
- **AWS Data Wrangler** → Simplifies Pandas-based data transformations in AWS.
- **Amazon SageMaker Processing** → Preprocessing for ML workloads.

---

### **2. Choosing the Right Data Transformation Solution**
| **Use Case**                         | **Best Transformation Service** |
|-------------------------------------|------------------------------|
| Batch ETL Processing                | **AWS Glue**                 |
| Large-scale distributed ETL         | **Amazon EMR**               |
| Real-time event processing          | **Kinesis Data Analytics**   |
| Serverless transformations          | **AWS Lambda**               |
| SQL-based transformation on S3 data | **Redshift Spectrum**        |
| Python-based Pandas transformation  | **AWS Data Wrangler**        |
| ML-specific data processing         | **SageMaker Processing**     |

---

### **3. Batch vs. Real-Time Transformation**
#### **Batch Processing**
- **AWS Glue** (recommended for ETL) or **Amazon EMR** (for big data).
- Used for **large, periodic transformations** (e.g., daily batch jobs).
- **Example**: Cleaning raw customer purchase data in S3 before loading to Redshift.

#### **Real-Time Transformation**
- **AWS Lambda** → Low-latency transformations for event-driven architectures.
- **Kinesis Data Analytics** → SQL-based real-time streaming transformations.
- **Amazon EMR with Spark Streaming** → Large-scale stream processing.
- **Example**: Processing IoT sensor data in near real-time.

- **Rule of Thumb**: Use **batch for scheduled processing**, **real-time for streaming transformations**.

---

### **4. Common Data Transformation Techniques**
- **Data Cleaning** → Handling missing values, deduplication.
- **Data Normalization** → Scaling numerical values for ML.
- **Feature Engineering** → Creating new features for ML models.
- **Data Aggregation** → Summarizing data at different levels (daily, monthly).
- **Schema Mapping** → Converting data formats (e.g., CSV to Parquet).
- **Data Enrichment** → Combining data from multiple sources.

---

### **5. Example Transformation Pipelines**
#### **1. Batch ETL Pipeline**
- **Source**: S3, databases.
- **Transformation**: AWS Glue (PySpark, Python).
- **Storage**: Amazon Redshift, S3.

#### **2. Real-Time Transformation Pipeline**
- **Source**: IoT devices, logs, clickstream data.
- **Transformation**: Kinesis Data Analytics, AWS Lambda.
- **Storage**: Amazon OpenSearch, DynamoDB, S3.

#### **3. ML Data Processing Pipeline**
- **Source**: S3, Data Lake.
- **Transformation**: SageMaker Processing (Pandas, Scikit-learn).
- **Storage**: Amazon Feature Store, S3.

---

### **6. Security & Cost Considerations**
- **Security**:
  - Use **IAM Roles & Policies** to restrict access.
  - Use **KMS encryption** for securing transformed data.
- **Cost Optimization**:
  - **Glue vs EMR**: Use Glue for serverless, EMR for massive-scale processing.
  - **Lambda vs Kinesis Analytics**: Use Lambda for lightweight transformations, Kinesis Analytics for continuous data streams.

---

## **Exam Rules of Thumb**
- **Batch ETL Processing** → Use **AWS Glue** (serverless) or **EMR** (big data).
- **Real-Time Transformations** → Use **Kinesis Data Analytics** for streaming, **AWS Lambda** for event-driven transformations.
- **SQL-Based Transformations on S3** → Use **Redshift Spectrum**.
- **For Python-based Transformations** → Use **AWS Data Wrangler**.
- **For ML Preprocessing** → Use **SageMaker Processing**.
- **Secure Transformations** → Use **IAM, VPC Endpoints, and KMS Encryption**.

# Domain 2: Exploratory Data Analysis
## Task Statement 2.1: Sanitize and prepare data for modeling.
## Task Statement 2.2: Perform feature engineering.
## Task Statement 2.3: Analyze and visualize data for ML.



# Domain 3: Modeling
## Task Statement 3.1: Frame business problems as ML problems.
## Task Statement 3.2: Select the appropriate model(s) for a given ML problem.
## Task Statement 3.3: Train ML models.
## Task Statement 3.4: Perform hyperparameter optimization.
## Task Statement 3.5: Evaluate ML models.


# Domain 4: Machine Learning Implementation and Operations
## Task Statement 4.1: Build ML solutions for performance, availability, scalability,
resiliency, and fault tolerance.
## Task Statement 4.2: Recommend and implement the appropriate ML services and
features for a given problem.
## Task Statement 4.3: Apply basic AWS security practices to ML solutions
## Task Statement 4.4: Deploy and operationalize ML solutions.





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


