---
title: "Generalized Models vs Specialized Models"
date: 2023-09-05T11:50:57Z
draft: False
ShowToc: true
tags: [Model Architecture,ML Design]
math: true
url: /posts/generalvsspecializedmodels/
---

# TLDR;

This blogpost focusses on ML Design and Architecture and tries to give some hints for deciding between one generalized and multiple specialized models for the same business requriement and dataset. *Consider it a nudge to dive deeper into the topic*

# Why should i care?

Transparency in machine learning is crucial for business stakeholders because it fosters trust and informed decision-making. Business Stakeholders need to understand not only the potential benefits but also the limitations and risks associated with machine learning models.

As Data scientists we play an important role in this process, as they bridge the gap between complex algorithms and business objectives. Educating stakeholders about the inner workings of machine learning algorithms, their inputs, outputs, and potential biases, empowers them to make well-informed decisions, manage expectations, and mitigate risks effectively.

> But this customer segment has way higher churn, shouldn't the model focus only on this segment only?

This is one of the examples when you as a ML Engineer need to shed some light on your choices and the "machine room" aka the used algorithm to make clear how the model works.

# UseCase

We are focussing on a "Bread and Butter" Binary Classification Model for Churn in the Telecommunications sector and try to make a comparison between the two approaches.

"Churn" refers to the phenomenon where customers switch from one telecommunications service provider to another or discontinue their subscription altogether. It is a significant metric and concern for telecommunications companies because retaining existing customers is often more cost-effective than acquiring new ones.

A binary classification model is a type of machine learning model used to classify data into one of two categories: a positive class (labeled as *1*) or a negative class (labeled as *0*). It learns from labeled data during training and uses features to make predictions. The model is evaluated based on metrics like accuracy, precision, recall, and F1-score and can be applied to various tasks, such as spam detection, disease diagnosis, and sentiment analysis. It's a fundamental tool for making decisions with two possible outcomes.

# General Models vs Specialized Models

Suppose we have a dataset which consist of a matrix of predictors (called *X*) and a target variable (called *y*). *X* contains *n* columns(features) that could be used to segment the dataset. E.g. a dataset can be segmented by age, product, sales channel, network experience...

So what are the two approaches we want to look into:

> **General Model**
>
> One main model is fitted on the whole training set including all segments, then the performance is measured on the test set.

> **Specialized Model**
>
> The dataset is splitted into each segment and for each of these subsets a unique model is fitted. This means that we repeat training and testing for the number of *k* segments.

## Intuition

Using specialized models has obviously some practical disadvantages as you need to do some tasks *k* times which leads to
- higher complexity
- higher effort on maintenance
- almost redundant processes

**So why should anyone favour specialized models at any point in time?**

The prejudice against general models is as following: Advocates for specialized models argue that a single, all-encompassing model might lack precision within a specific subset, as it would have learned the characteristics of various other subsets. 

This intuition was in my opinion built on top of the assumption that all Machine Learning Models work similary to simple models like e.g. linear regression.
Linear regression assumes that there is a linear relationship between the independent variable(s) *X* and the dependent variable *y*. This means that the relationship can be approximated by a straight line. While it assumes linearity, it can be extended to capture more complex relationships by introducing higher-order terms or using more advanced regression techniques like polynomial regression or multiple linear regression when dealing with multiple independent variables.
Still it lacks accuracy if the dataset contains different behaviours.

The intuition does not neccessarily hold true for the de facto standard algorithm for tabular data: Boosted Tree Models like XGBoost,LightGBM or CatBoost.

# The type of model counts

Boosted tree algorithms, outperform linear models like linear regression in modeling complexity. The main advantages above linear models which are relevant to defuse the intuition against general models is their ability to:

- **Handle Non-Linearity:** Boosted trees capture non-linear relationships, unlike linear regression, which assumes linearity.

- **Ensemble Learning:** They use ensemble learning to combine multiple decision trees, enabling them to create complex models.

- **Feature Interactions:** Boosted trees automatically detect and model feature interactions, which linear regression struggles with.

- **Model Flexibility:** Boosted trees adapt to data complexity by adding more trees to the ensemble.

This is the main reason why there is no theoretical reason to prefer several specialized models over one general model.
Other adavantages are:

- **Robustness to Outliers:** They are more robust to outliers and noisy data, making them suitable for real-world scenarios.

- **Handling Heterogeneous Data:** They handle various data types, including categorical features, without extensive preprocessing.

- **Fewer Assumptions:** Unlike linear regression, they have fewer rigid assumptions, making them versatile in diverse datasets.

# And what about the data

When looking at the data and keeping in mind the capabilities of non-linear models we should always prefer general model over specialized models when there is some similarity across the segments composing the dataset.

As the segments diverge further from one another, the benefits of employing a universal model diminish progressively.
If the segments are entirely dissimilar, the disparity between the two approaches should converge to zero. 

But what would that mean when looking at our Real-Life UseCase? Will customer segments be completely different. Shall we assume that the behaviour of people based on age, product, sales channel, network experience is completely different?

As you already can see this is a rhetoric question because you cannot assume completely different behaviour for a more or less homogenous group.

# I want proof

The former thoughts are based on two very nice blogposts by Samuele Mazzanti which are also referenced below.

He did the math and calculated based on two experiments whether a general model outperforms specialized models.

On average the general model outperforms the specialized models. He did a short stress test with completely diverged groups and fake data and even then the general model performed only 0.53% AUC worse than the specialized model.

# Conclusion

Start with a general model, check your data and use specialized models only in rare cases.

# Ressources

![Samuele Mazzanti: What Is Better: One General Model or Many Specialized Models?](https://towardsdatascience.com/what-is-better-one-general-model-or-many-specialized-models-9500d9f8751d)

![Samuele Mazzanti: The Unreasonable Effectiveness of General Models](https://towardsdatascience.com/the-unreasonable-effectiveness-of-general-models-b4e822eaeb27#:~:text=The%20mean%20difference%20between%20the,generally%20outperformed%20the%20general%20model.)


