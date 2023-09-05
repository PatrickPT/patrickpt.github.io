---
title: "Generalized Models vs Specialized Models"
date: 2023-09-05T11:50:57Z
draft: True
ShowToc: true
tags: [Model Architecture,ML Design]
math: true
url: /posts/generalvsspecializedmodels/
---

# TLDR;

This blogpost focusses on ML Design and Architecture and tries to give some hints for deciding between one generalized and multiple specialized models for the same business requriement and dataset.

# What's the catch?

Transparency in machine learning is crucial for business stakeholders because it fosters trust and informed decision-making. Business Stakeholders need to understand not only the potential benefits but also the limitations and risks associated with machine learning models.

As Data scientists we play an important role in this process, as they bridge the gap between complex algorithms and business objectives. Educating stakeholders about the inner workings of machine learning algorithms, their inputs, outputs, and potential biases, empowers them to make well-informed decisions, manage expectations, and mitigate risks effectively.

> But this customer segment has way higher churn, shouldn't the model focus only on this segment only?

This is one of the examples when you as a ML Engineer need to shed some light on your choices and the "machine room" aka the used algorithm to make clear how the model works.

# General Models vs Specialized Models

Suppose we have a dataset which consist of a matrix of predictors (called X) and a target variable (called y). X contains n columns that could be used to segment the dataset (suppose we would have a customer set segmented by age).

So what are the two approaches we want to look into:

## 

# Ressources

![](https://towardsdatascience.com/what-is-better-one-general-model-or-many-specialized-models-9500d9f8751d)



