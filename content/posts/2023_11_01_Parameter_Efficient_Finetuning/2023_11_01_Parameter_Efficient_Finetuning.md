---
title: "Parameter Efficient Finetuning"
date: 2023-11-01T18:30:57Z
draft: True
ShowToc: true
tags: [Model Architecture,AI Design, LLM, GenAI]
math: true
url: /posts/peft/
images:
    #- /posts/

---

# TL;DR

Parameter Efficient Fine-Tuning is a technique that aims to reduce computational and storage resources during the fine-tuning of Large Language Models.

# Why should i care?

Fine-tuning is a common technique used to enhance the performance of large language models. Essentially, fine-tuning involves training a pre-trained model on a new, similar task. It has become a crucial step in model optimization. However, when these models consist of billions of parameters, fine-tuning becomes computational and storage heavy, leading to the development of Parameter Efficient Fine-Tuning methods.

Parameter Efficient Fine-Tuning is a technique that aims to reduce computational and storage resources during the fine-tuning process. Rather than tuning all parameters, these methods strategically select and fine-tune a fraction of them. The rest of the parameters are frozen or updated with a lower precision format, saving memory and computational requirements. This technique also mitigates the risk of overfitting, enhances learning efficiency, and allows for more adaptability in applying large language models across a variety of tasks.

# Use Parameter Efficient Finetuning

## LoRA: An Introduction

The Layerwise Optimized Rank-Adaptive (LoRA) method by Google Research is a prime example of Parameter Efficient Fine-Tuning. The technique is based on a simple premise: creating new, low-rank parameters that can be fine-tuned, leaving the original parameters of the pre-trained model unchanged.

LoRA introduces a small number of additional parameters, allows the base model to maintain its optimized parameters, and ensures that most computational resources can be devoted to the new parameters. As such, it provides a fine-tuning method that not only maintains the base model's strength but also adapts effectively to new tasks.

## QLoRA: Taking it One Step Further

Compressing model parameters even more, QLoRA – or Quantized LoRA – takes the concepts of LoRA and applies a quantization technique. Instead of storing these new low-rank parameters in standard floating-point format, QLoRA stores them in a lower precision format, such as INT8 or INT4.

This method drastically reduces the number of bits required to store these parameters. As a result, QLoRA offers a significant reduction in the memory and computational requirements, vastly improving parameter efficiency during the fine-tuning process.

## Exploring Other Methods

Other parameter efficient fine-tuning methods like AdaFit, Piggyback, and SpotTune also apply a similar principle. They work by creating a small separate mask or layer for fine-tuning while not disturbing the original model's parameters. These methods lean on the side of versatility and efficiency, offering the benefits of large language model fine-tuning but at a lower computational and storage cost, ideal for resource-constrained environments or real-world applications.

## Conclusion

The challenges associated with fine-tuning large language models, such as high memory requirements and computational burden, have prompted researchers to create innovative solutions like LoRa, QLoRa, AdaFit, and other parameter efficient fine-tuning methods. By creating a balance between computational power and performance, these techniques make the power of large language models more accessible and applicable to an array of tasks. As natural language processing continues to advance, the importance of such computation-friendly and efficient methods will undoubtedly continue to grow.

# Benefits

# Pitfalls
 
# Conclusion

# Resources

[https://www.leewayhertz.com/parameter-efficient-fine-tuning/] [Blogpost]