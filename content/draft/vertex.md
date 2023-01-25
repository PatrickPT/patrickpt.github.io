---
title: "PUBLIC DRAFT/Road to GCP Professional Machine Learning Engineer Exam"
date: 2023-01-25T21:01:23+01:00
draft: false
showToc: true
TocOpen: false
math: true
tags: [GCP,MLOPS,ML Engineer]
url: /draft/vertex/
---

# Ressources

## Official

[Google Documentation](https://cloud.google.com/vertex-ai/docs?hl=en)

[Google Exam Curriculum](https://cloud.google.com/certification/guides/machine-learning-engineer?hl=en)

[Google Cloudskillboost](https://www.cloudskillsboost.google/paths/17)

[Google ML Crash Course](https://developers.google.com/machine-learning/crash-course?hl=en)

[Google Example Questions](https://docs.google.com/forms/d/e/1FAIpQLSeYmkCANE81qSBqLW0g2X7RoskBX9yGYQu-m1TtsjMvHabGqg/viewform)

## Other ressources
| Helpful | Published | Title/Link | Author |
| :-: | :---:         |     :---      |          :--- |
| 10 |  2022/11 | [Journey to PMLE(O'Reilly Paywall)](https://learning.oreilly.com/library/view/journey-to-become/9781803233727/B18333_01.xhtml#_idParaDest-22) | Dr. Logan Song |
| 7 |  2022/12 | [Awesome GCP Certifications Repo](https://github.com/sathishvj/awesome-gcp-certifications/blob/master/professional-machine-learning-engineer.md) | Satish Vijai |
| 9 |  2022/12 | [Passing the Google cloud professional machine learning engineer exam](https://medium.com/@hilliao/passing-the-google-cloud-professional-machine-learning-engineer-exam-ee5109ad77f4) | Hil Liao |
| 5 |  2022/12 | [How to clear GCP PMLE Exam?](https://medium.com/@techwithshadab/how-to-clear-google-cloud-professional-machine-learning-exam-3beeed012c48) | Shadab Hussain |
| 3 |  2022/03 | [How I passed the Google Cloud PMLEexam (Vertex AI)](https://medium.com/@joshcx/how-i-passed-the-google-cloud-professional-machine-learning-engineer-exam-vertex-ai-484c7863bbac) | Joshua Tan |
| 2 |  2022/01 | [How to prepare for the GCP PMLEexam](https://towardsdatascience.com/how-to-prepare-for-the-gcp-professional-machine-learning-engineer-exam-b1c59967355f) | Gabriel Cassimiro|
| 10 |  2022/01 | [A cromprehensive Study Guide](https://towardsdatascience.com/a-comprehensive-study-guide-for-the-google-professional-machine-learning-engineer-certification-1e411db4d2cf) | Jeffrey Luppes |
| 6 |  2021/03 | [Study guide to ace Google Cloud Certification on PMLE](https://medium.com/analytics-vidhya/study-guide-to-ace-google-cloud-certification-on-professional-machine-learning-engineer-2d6a05f9fbba) | Rahul Pandey |
| 2 |  2021/01 | [Learning Notes](https://github.com/sehgalnamit/Preparing-for-Google-cloud-professional-machine-learning-engineer-/blob/main/GCP_ML_Professional_Prepare.docx) | Sehgal Namit |

# Notes

## Key takeaways

- A significant amount of knowledge covered in the exam also came from Google’s machine learning crash course.

## Documentation hints in short
Read the documentation (don’t focus on the code, but more on when to use which tool). Reading the whole GCP documentation can take forever so, focus on these specific areas:
- ML APIs — Vision, Natural Language , Video and Speech-to-text (understand what each one of them can do and also what they can’t)
- AutoML (learn when you should use AutoML instead of ML APIs)
- Vertex (this is the most important part. Focus on how to improve performance, how to use accelerators such as TPUs and GPUs, how to do distributed training and serving and the different available tools, such as the What-If Tool)
- Recommendations AI (there are 3 model types. Learn when to use each of them)
- TPUs (know when you should use them, and how)
- TensorFlow (do not memorize code, but how to improve performance)
- BigQuery ML (learn what are all the available algorithms) 

# Curriculum commented

**TODO: Do i really wanna do this?**

## Section 1: Framing ML problems
### 1.1 Translating business challenges into ML use cases.
- Choosing the best solution (ML vs. non-ML, custom vs. pre-packaged [e.g., AutoML, Vision API]) based on the business requirements 
- Defining how the model output should be used to solve the business problem
- Deciding how incorrect results should be handled
- Identifying data sources (available vs. ideal)

### 1.2 Defining ML problems.
- Problem type (e.g., classification, regression, clustering)
- Outcome of model predictions
- Input (features) and predicted output format

### 1.3 Defining business success criteria.
- Alignment of ML success metrics to the business problem
- Key results
- Determining when a model is deemed unsuccessful

### 1.4 Identifying risks to feasibility of ML solutions. 
- Assessing and communicating business impact
- Assessing ML solution readiness
- Assessing data readiness and potential limitations
- Aligning with Google’s Responsible AI practices (e.g., different biases)

## Section 2: Architecting ML solutions
### 2.1 Designing reliable, scalable, and highly available ML solutions.
- Choosing appropriate ML services for the use case (e.g., Cloud Build, Kubeflow)
- Component types (e.g., data collection, data management)
- Exploration/analysis
- Feature engineering
- Logging/management
- Automation
- Orchestration
- Monitoring
- Serving

### 2.2 Choosing appropriate Google Cloud hardware components.
- Evaluation of compute and accelerator options (e.g., CPU, GPU, TPU, edge devices) 

### 2.3 Designing architecture that complies with security concerns across sectors/industries. 
Considerations include:
- Building secure ML systems (e.g., protecting against unintentional exploitation of data/model, hacking)
- Privacy implications of data usage and/or collection (e.g., handling sensitive data such as Personally Identifiable Information [PII] and Protected Health Information [PHI])

## Section 3: Designing data preparation and processing systems
### 3.1 Exploring data (EDA).
- Visualization
- Statistical fundamentals at scale
- Evaluation of data quality and feasibility
- Establishing data constraints (e.g., TFDV)

### 3.2 Building data pipelines.
- Organizing and optimizing training datasets
- Data validation
- Handling missing data
- Handling outliers
- Data leakage

### 3.3 Creating input features (feature engineering).
- Ensuring consistent data pre-processing between training and serving
- Encoding structured data types
- Feature selection
- Class imbalance
- Feature crosses
- Transformations (TensorFlow Transform)

## Section 4: Developing ML models
#### 4.1 Building models.
- Choice of framework and model
- Modeling techniques given interpretability requirements
- Transfer learning
- Data augmentation
- Semi-supervised learning
- Model generalization and strategies to handle overfitting and underfitting

### 4.2 Training models.
- Ingestion of various file types into training (e.g., CSV, JSON, IMG, parquet or databases, Hadoop/Spark)
- Training a model as a job in different environments
- Hyperparameter tuning
- Tracking metrics during training
- Retraining/redeployment evaluation

### 4.3 Testing models.
- Unit tests for model training and serving
- Model performance against baselines, simpler models, and across the time dimension
- Model explainability on Vertex AI

### 4.4 Scaling model training and serving.
- Distributed training
- Scaling prediction service (e.g., Vertex AI Prediction, containerized serving)

## Section 5: Automating and orchestrating ML pipelines
### 5.1 Designing and implementing training pipelines.
- Identification of components, parameters, triggers, and compute needs (e.g., Cloud Build, Cloud Run)
- Orchestration framework (e.g., Kubeflow Pipelines/Vertex AI Pipelines, Cloud Composer/Apache Airflow)
- Hybrid or multicloud strategies
- System design with TFX components/Kubeflow DSL 

### 5.2 Implementing serving pipelines.
- Serving (online, batch, caching)
- Google Cloud serving options
- Testing for target performance
- Configuring trigger and pipeline schedules

### 5.3 Tracking and auditing metadata.
- Organizing and tracking experiments and pipeline runs
- Hooking into model and dataset versioning
- Model/dataset lineage

## Section 6: Monitoring, optimizing, and maintaining ML solutions
### 6.1 Monitoring and troubleshooting ML solutions. 
Considerations include:
- Performance and business quality of ML model predictions
- Logging strategies
- Establishing continuous evaluation metrics (e.g., evaluation of drift or bias)
- Understanding Google Cloud permissions model
- Identification of appropriate retraining policy
- Common training and serving errors (TensorFlow)
- ML model failure and resulting biases
### 6.2 Tuning performance of ML solutions for training and serving in production. 
Considerations include:
- Optimization and simplification of input pipeline for training
- Simplification techniques 

