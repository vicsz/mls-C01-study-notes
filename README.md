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
## **Task Statement 2.1: Sanitize and Prepare Data for Modeling**

### **1. Data Sanitization & Cleaning**
- **Goal**: Ensure the dataset is clean, complete, and ready for ML modeling.
- **Common Issues**:
  - **Missing Values** → Use imputation (mean, median, mode) or remove rows.
  - **Duplicate Data** → Remove identical rows or deduplicate based on criteria.
  - **Outliers** → Use Z-score, IQR method, or visualization techniques to detect them.
  - **Incorrect Data Types** → Convert categorical data to numerical (one-hot encoding, label encoding).
  - **Inconsistent Formatting** → Standardize date formats, units of measurement.
- **AWS Tools**:
  - **AWS Glue** → Cleans data using PySpark.
  - **Amazon Data Wrangler** → Prepares data using Pandas-like transformations.
  - **Amazon EMR** → Scales data processing for large datasets.
  - **Amazon Athena** → Queries and filters data directly from S3.

---

### **2. Feature Engineering**
- **Feature Engineering** → Creating new meaningful features to improve model accuracy.
- **Common Techniques**:
  - **Normalization & Standardization** → Scale numerical data (MinMaxScaler, StandardScaler).
  - **One-Hot Encoding** → Convert categorical variables into binary vectors.
  - **Feature Selection** → Remove redundant or irrelevant features (e.g., using PCA).
  - **Feature Extraction** → Transform raw data into features (e.g., text embeddings).
- **AWS Tools**:
  - **Amazon SageMaker Feature Store** → Stores reusable features for ML models.
  - **SageMaker Processing** → Runs feature engineering scripts at scale.
  - **Amazon Data Wrangler** → Simplifies feature transformation with Pandas-like operations.

---

### **3. Handling Imbalanced Datasets**
- **Problem**: One class dominates over another, causing biased models.
- **Solutions**:
  - **Oversampling** → Duplicate underrepresented class instances (SMOTE).
  - **Undersampling** → Remove some instances from the dominant class.
  - **Class Weights** → Adjust model loss function to penalize misclassification of the minority class.
- **Rule of Thumb**: Use **SMOTE for small datasets, class weighting for large datasets**.

---

### **4. Data Labeling & Annotation**
- **Use Case**: Preparing labeled data for supervised learning.
- **AWS Tools**:
  - **Amazon SageMaker Ground Truth** → Human-in-the-loop data labeling.
  - **Amazon Augmented AI (A2I)** → Automates human review for ML predictions.
- **Rule of Thumb**: Use **Ground Truth for large-scale data labeling, A2I for selective human review**.

---

### **5. Data Splitting for Training & Validation**
- **Train/Test Split**:
  - **Standard Split** → **80/20 or 70/30** for general ML tasks.
  - **Time-Series Split** → Use **rolling windows** to prevent data leakage.
  - **Stratified Sampling** → Ensures class balance in classification tasks.
- **Cross-Validation**:
  - **K-Fold Cross-Validation** → Reduces bias by splitting data into multiple folds.
  - **Leave-One-Out (LOO) Cross-Validation** → Used for very small datasets.
- **Rule of Thumb**: Use **K-Fold (default), rolling windows for time-series data**.

---

### **6. Security & Compliance in Data Preparation**
- **Data Anonymization** → Remove personally identifiable information (PII).
- **Data Encryption**:
  - **At Rest** → Use **AWS KMS for S3, Redshift, RDS encryption**.
  - **In Transit** → Use **TLS encryption for network security**.
- **Data Access Control**:
  - **IAM Policies & Lake Formation** → Restrict access.
  - **Amazon Macie** → Detects sensitive data (PII, credit card info).
- **Rule of Thumb**: Use **Macie for PII detection, KMS for encryption, IAM/Lake Formation for access control**.

---

## **Exam Rules of Thumb**
- **For missing values** → Use **mean/median imputation**.
- **For categorical data** → Use **one-hot encoding** (low cardinality) or **label encoding** (high cardinality).
- **For feature scaling** → Use **MinMaxScaler** (if data range is known) or **StandardScaler** (if normally distributed).
- **For class imbalance** → Use **SMOTE for small datasets, class weights for large datasets**.
- **For data splitting** → Use **80/20 for ML, rolling windows for time-series**.
- **For data labeling** → Use **SageMaker Ground Truth**.
- **For data security** → Use **Macie for PII detection, KMS for encryption**.

## **Task Statement 2.2: Perform Feature Engineering**

### **1. What is Feature Engineering?**
- **Feature Engineering** involves creating, transforming, and selecting features to improve model performance.
- **Goal**: Enhance the predictive power of the model by providing meaningful representations of data.

---

### **2. Common Feature Engineering Techniques**
| **Technique**            | **Description** | **Example Use Case** |
|-------------------------|----------------|----------------------|
| **Normalization** | Scales data between 0 and 1. | Pixel values in image processing. |
| **Standardization** | Centers data to mean 0, std 1. | Financial transactions, sensor data. |
| **One-Hot Encoding** | Converts categorical variables into binary vectors. | "Color: Red, Green, Blue" → 3 columns. |
| **Label Encoding** | Assigns unique numeric values to categories. | "Small = 0, Medium = 1, Large = 2". |
| **Feature Selection** | Removes irrelevant or redundant features. | PCA, Mutual Information. |
| **Feature Extraction** | Derives new features from raw data. | TF-IDF for text, embeddings for NLP. |
| **Binning** | Groups continuous values into buckets. | Age ranges: 0-18, 19-35, 36-50, 51+. |
| **Polynomial Features** | Creates interaction terms between features. | \(X^2, X_1 \times X_2\) in regression models. |
| **Time-Series Features** | Extracts patterns from time-based data. | Weekday, seasonality, lag variables. |

---

### **3. Feature Scaling Techniques**
#### **Normalization (MinMax Scaling)**
- Rescales data to a range of **0 to 1**.
- **Use Case**: When feature distributions vary widely.

#### **Standardization (Z-score Normalization)**
- Centers data around **mean 0, variance 1**.
- **Use Case**: When data follows a normal distribution.

- **Rule of Thumb**: Use **MinMaxScaler for bounded values**, **StandardScaler for normally distributed data**.

---

### **4. Handling Categorical Features**
#### **One-Hot Encoding**
- Converts categorical variables into multiple binary columns.
- **Best for low-cardinality categorical features**.

#### **Label Encoding**
- Assigns numeric labels to categories.
- **Best for high-cardinality categorical features**.

- **Rule of Thumb**: Use **one-hot encoding for categorical data with few categories**, **label encoding for high-cardinality categorical features**.

---

### **5. Feature Selection Techniques**
#### **Filter Methods**:
- Uses **statistical tests** to select the most relevant features.
- Examples:
  - **Chi-Square Test** → Categorical features.
  - **Mutual Information** → Finds the most informative features.
  - **Variance Threshold** → Removes low-variance features.

#### **Wrapper Methods**:
- Iteratively adds or removes features while training models.
- Examples:
  - **Recursive Feature Elimination (RFE)** → Eliminates least important features.
  - **Forward/Backward Feature Selection** → Adds/removes features based on model performance.

#### **Embedded Methods**:
- Feature selection occurs during model training.
- Examples:
  - **LASSO Regression** → Uses L1 regularization to shrink feature weights.
  - **Tree-Based Models** → Decision trees, Random Forests, and XGBoost rank feature importance.

- **Rule of Thumb**: Use **Filter Methods for preprocessing**, **Wrapper Methods for model tuning**, and **Embedded Methods when training models**.

---

### **6. Feature Engineering for Time-Series Data**
- **Lag Features** → Previous values as new features (e.g., "Sales 7 days ago").
- **Rolling Window Statistics** → Moving averages, min/max values over a time window.
- **Seasonality Features** → Extracting day, month, year, weekday, holidays.
- **Fourier Transforms** → Captures periodicity in time-series data.

- **Rule of Thumb**: Use **lag features for short-term forecasting**, **rolling stats for trend detection**, and **seasonality indicators for capturing cyclic patterns**.

---

### **7. Feature Engineering for NLP (Text Data)**
- **TF-IDF (Term Frequency-Inverse Document Frequency)** → Measures word importance in a document.
- **Word Embeddings (Word2Vec, GloVe, BERT)** → Converts text into numerical vectors.
- **Text Cleaning** → Lowercasing, stopword removal, stemming, lemmatization.
- **N-Grams** → Captures sequences of words (e.g., "New York" as a unit).

- **Rule of Thumb**: Use **TF-IDF for traditional NLP models**, **Word2Vec/BERT for deep learning models**.

---

### **8. Automating Feature Engineering**
- **Amazon SageMaker Feature Store** → Centralized repository for storing & sharing features.
- **AWS Glue** → ETL processing for feature transformation.
- **AWS Data Wrangler** → Pandas-like transformations in AWS.

---

## **Exam Rules of Thumb**
- **For feature scaling** → Use **MinMaxScaler** (if data range is known) or **StandardScaler** (if normally distributed).
- **For categorical encoding** → Use **One-Hot Encoding for low-cardinality**, **Label Encoding for high-cardinality**.
- **For feature selection** → Use **PCA, RFE, or Mutual Information**.
- **For time-series forecasting** → Use **lag features, rolling windows, and seasonality indicators**.
- **For NLP** → Use **TF-IDF for simple models, Word Embeddings for deep learning**.
- **For automation** → Use **SageMaker Feature Store, AWS Glue, or Data Wrangler**.

## **Task Statement 2.3: Analyze and Visualize Data for ML**

### **1. Importance of Data Analysis and Visualization**
- Helps identify **patterns, trends, correlations, and anomalies** in the dataset.
- Ensures **data quality** by detecting **missing values, outliers, and skewness**.
- Assists in **feature selection and engineering** for better ML model performance.

---

### **2. Key Statistical Measures for Data Analysis**
| **Metric**          | **Description**                                  | **Use Case** |
|--------------------|------------------------------------------------|-------------|
| **Mean (Average)** | Central tendency of the data.                  | Salary estimation, revenue trends. |
| **Median**        | Middle value of the dataset (less sensitive to outliers). | House price predictions. |
| **Mode**         | Most frequently occurring value.                 | Categorical data analysis. |
| **Standard Deviation** | Measure of data dispersion.                 | Identifying volatility in stock prices. |
| **Skewness**     | Measures asymmetry of data distribution.         | Detecting fraud in financial transactions. |
| **Kurtosis**     | Measures tail heaviness of distribution.         | Risk assessment in finance. |
| **Correlation (Pearson/Spearman)** | Measures relationship between features. | Feature selection in ML models. |

- **Rule of Thumb**: Use **mean for normally distributed data, median for skewed data, correlation for feature selection**.

---

### **3. Handling Missing Data**
- **Identify missing values**:
  - Percentage of missing values per column.
- **Solutions**:
  - **Remove rows/columns** (if missing values are few).
  - **Impute missing values**:
    - **Numerical** → Mean, median, mode, interpolation.
    - **Categorical** → Most frequent value or a placeholder category ("Unknown").
- **Rule of Thumb**: **Drop missing data if <5% missing**, **impute using mean/median if continuous**, **use mode for categorical data**.

---

### **4. Detecting and Handling Outliers**
- **Outlier Detection Methods**:
  - **Standard Deviation Method** → If value >3 standard deviations from mean.
  - **Interquartile Range (IQR) Method** → Detects values outside 1.5×IQR.
  - **Z-score** → Identifies extreme values.
- **Handling Outliers**:
  - **Remove** if they result from data entry errors.
  - **Transform** using log transformation if skewed.
  - **Cap** extreme values using percentile-based trimming.
- **Rule of Thumb**: **Use IQR for skewed data, Z-score for normally distributed data**.

---

### **5. Data Visualization Techniques**
| **Visualization Type**   | **Use Case**                                   | **Best For** |
|-------------------------|-----------------------------------------------|-------------|
| **Histogram**          | Shows data distribution.                      | Detecting skewness, outliers. |
| **Boxplot**            | Identifies outliers and spread of data.        | Visualizing numerical data. |
| **Scatter Plot**       | Shows relationship between two variables.      | Identifying correlations. |
| **Pairplot**           | Shows pairwise relationships between features. | Feature selection. |
| **Heatmap**            | Displays correlation between variables.        | Feature importance analysis. |
| **Line Chart**         | Shows trends over time.                        | Time-series analysis. |
| **Bar Chart**          | Compares categorical data.                     | Frequency distribution. |
| **Pie Chart**          | Shows proportion of categories.                | Distribution comparison. |

- **Rule of Thumb**: Use **histograms for distributions, boxplots for outliers, scatter plots for relationships, and heatmaps for correlations**.

---

### **6. Analyzing Feature Relationships**
#### **Correlation Analysis**
- **Pearson Correlation** → Measures linear relationships between numerical variables.
- **Spearman Correlation** → Measures rank-order relationships (best for non-linear data).
- **Use Case**:
  - High correlation (close to 1 or -1) → Features may be redundant.
  - Low correlation (close to 0) → Features are independent.

#### **Feature Importance Analysis**
- **Tree-based Models** → Random Forest, XGBoost provide feature importance scores.
- **Mutual Information** → Measures relevance of each feature to the target variable.

- **Rule of Thumb**: **Drop highly correlated features (>0.8 correlation), prioritize top features using tree-based models**.

---

### **7. Time-Series Data Analysis**
- **Moving Averages** → Smoothens trends over time.
- **Seasonality Analysis** → Extracts repeating patterns (e.g., daily, weekly trends).
- **Autocorrelation** → Checks dependency between time steps.
- **Lag Features** → Uses past observations as new features.

- **Rule of Thumb**: **Use moving averages for trend analysis, seasonality decomposition for recurring patterns, and lag features for forecasting**.

---

### **8. AWS Tools for Data Analysis & Visualization**
| **AWS Service**               | **Use Case** |
|--------------------------------|-------------|
| **Amazon QuickSight**         | BI visualization for business data. |
| **Amazon SageMaker Data Wrangler** | Cleans, analyzes, and visualizes data. |
| **Amazon Athena**             | Runs SQL queries on S3 data. |
| **Amazon Glue DataBrew**      | No-code data exploration and cleaning. |
| **Amazon Redshift**           | SQL-based analysis of structured data. |

- **Rule of Thumb**: **Use QuickSight for BI dashboards, Data Wrangler for ML preprocessing, Athena for SQL-based exploration**.

---

## **Exam Rules of Thumb**
- **For detecting outliers** → Use **IQR for skewed data, Z-score for normal data**.
- **For handling missing data** → **Drop if <5% missing, otherwise impute**.
- **For feature selection** → **Use correlation analysis, tree-based models, mutual information**.
- **For visualization** → **Histograms for distributions, boxplots for outliers, heatmaps for correlations**.
- **For time-series** → **Use moving averages for trends, lag features for forecasting**.
- **For AWS tools** → **QuickSight for dashboards, Data Wrangler for ML prep, Glue DataBrew for no-code cleaning**.

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


