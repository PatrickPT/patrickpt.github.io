---
title: "Road to GCP Professional Machine Learning Engineer"
date: 2023-02-24T21:01:23+01:00
draft: false
summary: My learning journey to get certified
showToc: true
TocOpen: false
math: true
tags: [GCP,MLOPS,ML Engineer]
url: /posts/GCPMLE/
---

Today i passed the Google Professional Machine Learning Engineer Exam in a onsite test center in Duesseldorf. I prepared for it 4 weeks with different ressources. I have multiple years experience on Data Science and ML and about one year experience with ML on GCP.
Following article should summarize all sources which were helpful for me in preparation of the certification exam.
If you are interested in how the exam is actually happening there are plenty of other articles on medium or at other places. I just try to condense it to the minimum here.

![](/posts/2023_02_24_GCPMLE/images/gcpmle_certificate.jpg)

# Ressources

## Official

[Google Documentation](https://cloud.google.com/vertex-ai/docs?hl=en)

[Google Exam Curriculum](https://cloud.google.com/certification/guides/machine-learning-engineer?hl=en)

[Google Cloudskillboost](https://www.cloudskillsboost.google/paths/17)

## Most helpful Google ressources

[Google ML Crash Course](https://developers.google.com/machine-learning/crash-course?hl=en)

[Google Example Questions](https://docs.google.com/forms/d/e/1FAIpQLSeYmkCANE81qSBqLW0g2X7RoskBX9yGYQu-m1TtsjMvHabGqg/viewform)

[Best practices for implementing machine learning on Google Cloud](https://cloud.google.com/architecture/ml-on-gcp-best-practices#model-deployment-and-serving)

[Google Architecture for MLOps using TensorFlow Extended, Vertex AI Pipelines, and Cloud Build](https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build)

[Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml?hl=en)

[Google Cloud Architecture Center](https://cloud.google.com/architecture/ai-ml)

[Google Guidelines for ML Solutions](https://cloud.google.com/architecture/guidelines-for-developing-high-quality-ml-solutions)

## Other helpful ressources
| Helpful | Published | Title/Link | Author |
| :-: | :---:         |     :---      |          :--- |
| 10 |  - | [Google Cloud Cheat Sheet](https://googlecloudcheatsheet.withgoogle.com/) | - |
| 10 |  2022/11 | [Journey to PMLE(O'Reilly Paywall)](https://learning.oreilly.com/library/view/journey-to-become/9781803233727/B18333_01.xhtml#_idParaDest-22) | Dr. Logan Song |
| 7 |  2022/12 | [Awesome GCP Certifications Repo](https://github.com/sathishvj/awesome-gcp-certifications/blob/master/professional-machine-learning-engineer.md) | Satish Vijai |
| 9 |  2022/12 | [Passing the Google cloud professional machine learning engineer exam](https://medium.com/@hilliao/passing-the-google-cloud-professional-machine-learning-engineer-exam-ee5109ad77f4) | Hil Liao |
| 5 |  2022/12 | [How to clear GCP PMLE Exam?](https://medium.com/@techwithshadab/how-to-clear-google-cloud-professional-machine-learning-exam-3beeed012c48) | Shadab Hussain |
| 3 |  2022/03 | [How I passed the Google Cloud PMLEexam (Vertex AI)](https://medium.com/@joshcx/how-i-passed-the-google-cloud-professional-machine-learning-engineer-exam-vertex-ai-484c7863bbac) | Joshua Tan |
| 2 |  2022/01 | [How to prepare for the GCP PMLEexam](https://towardsdatascience.com/how-to-prepare-for-the-gcp-professional-machine-learning-engineer-exam-b1c59967355f) | Gabriel Cassimiro|
| 10 |  2022/01 | [A cromprehensive Study Guide](https://towardsdatascience.com/a-comprehensive-study-guide-for-the-google-professional-machine-learning-engineer-certification-1e411db4d2cf) | Jeffrey Luppes |
| 6 |  2021/03 | [Study guide to ace Google Cloud Certification on PMLE](https://medium.com/analytics-vidhya/study-guide-to-ace-google-cloud-certification-on-professional-machine-learning-engineer-2d6a05f9fbba) | Rahul Pandey |
| 2 |  2021/01 | [Learning Notes](https://github.com/sehgalnamit/Preparing-for-Google-cloud-professional-machine-learning-engineer-/blob/main/GCP_ML_Professional_Prepare.docx) | Sehgal Namit |

# Key takeaways

- A significant amount of knowledge covered in the exam also came from Google’s machine learning crash course.
- The questions on Cloudskillboost not neccessarily help with passing the exam
- You need a basic understanding of how MLOps works and which Google solutions supports which part in the process
- Think efficient - what is the easiest solution
- Think like a ML practicioner
- Read carefully the question and look for key-words: Cost, time, serverless, etc.
- Read carefully the answers and think of consequences of the solutions and try to identify the best solution
- Look into other sources on Medium of people who recently took the exam and try to identify new topics
- Search for Exam Dumps on deidcated sources like Examtopics. They may be helpful in preparation.

# Key topics

- Distributed Learning
- Imbalanced Data
- Efficiency (Scalability, Ease of Use, Reproducability)
- Speed (BQML vs Vertex Pipelines)
- Dataflow
- Triggers
- Metrics (Business vs ML)
- Where to do what (Inference on Device vs Pipeline)
- Privacy. How to set up a pipeline for the DLP(Quarantine Bucket)
- Tensorflow and Keras
