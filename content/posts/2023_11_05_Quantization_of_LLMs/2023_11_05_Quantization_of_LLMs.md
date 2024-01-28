---
title: "LLM Quantization in a nutshell"
date: 2024-01-28T09:30:57Z
draft: False
ShowToc: true
tags: [LLM, GenAI]
math: true
url: /posts/quantllm/

---

# TL;DR

This blogpost summarizes the buts and bolts of LLM quantization with llama.cpp.

# Introduction to Quantization
## The Technical Foundation of LLM Quantization
 
Quantization, in the context of machine learning, refers to the process of reducing the precision of a model's parameters, typically converting floating-point numbers to lower-bit representations. This has profound implications for model deployment, particularly in rendering sizable LLMs more accessible.
 
## Understanding Quantization
 
Quantization works by mapping the continuous range of floating-point values to a discrete set of levels. This is akin to reducing the bit depth in an audio file, where instead of infinite gradations, you have a limited number of steps. There are different strategies for quantization, such as post-training quantization, dynamic quantization, and quantization-aware training, each with its use cases and trade-offs.

### Post-Training Quantization
This approach involves reducing the precision of the weights of a model that has already been fully trained, which can be quickly accomplished without further training. While this method is simple to execute, it may slightly reduce the model's effectiveness due to a decrease in the precision of the weight values.
 
 
### Quantization-Aware Training
Contrary to the post-training method, this technique incorporates the lower precision conversion of weights during the model's training phase. This generally leads to better performance of the model but requires more computational resources. An example of a technique employed within this category is QLoRA.

## Optimization Techniques in Quantization
 
Once the quantization procedure is applied, the size of the model is significantly reduced. However, selecting the appropriate bit width is crucial—too few bits, and you may lose critical information; too many, and you compromise on efficiency gains. It’s also important to choose the right quantization technique. Techniques such as rounding or truncation can affect the performance of the quantized model.

# A Technical Examination
## Diving Deeper into llama.cpp
 
llama.cpp is a library written in C++ that allows users to perform inference on quantized LLaMA models. It provides tools to convert pre-trained models into a format suitable for inference on CPUs, particularly optimized for low-precision integer arithmetic, which dramatically reduces the computational load.
 
## Quantization in llama.cpp
 
Using llama.cpp involves specific steps—starting with installation of the library, obtaining a compatible LLaMA model, and then running the provided conversion script to quantize the model. The script typically outputs a new model file, which can be loaded using llama.cpp APIs for inference.
 
## Store quantized models

Quantized models are stored in the binary file format GGML or GGUF. "GG" refers to the initials of its originator (Georgi Gerganov).
GGML is a C library created for machine learning, notable for its unique binary format that facilitated the easy distribution of LLMs. This binary format was distinctive to GGML and provided a way to execute LLMs on a CPU, enhancing accessibility and use.
GGUF, represents an evolution from the original GGML format. It was developed to be more extensible and future-proof, while being lighter on RAM requirements.
GGUF is compatible with the llama.cpp library.

The reduced memory footprint with GGML/GGUF and the ability to offload certain layers of computation onto a GPU with GGML/GGUF accelerates inference speeds while managing models that would otherwise be too large for standard VRAM.

## Where to find models

The [Hugging Face Hub](https://huggingface.co/models) hosts a variety of pre-quantized models. You can find a collection of models employing diverse quantization techniques, offering choices that cater to specific requirements and scenarios.

# Conclusion
## Advantages of Quantization
 
Beyond the obvious benefits of reduced model size and computational resource requirements, there are other compelling reasons to quantize LLMs. For instance, lower memory bandwidth and power consumption are key factors for deployment on edge devices, and quantization can open up new possibilities for LLM applications in such constrained environments.
 
## Challenges and Considerations
 
Despite the efficiencies introduced through quantization, one must be mindful of potential accuracy loss. This necessitates a robust evaluation pipeline to assess the performance impact on tasks relevant to the language model's intended use. Furthermore, leveraging the right tooling and techniques to achieve optimal balance between size, efficiency, and accuracy is fundamental.
 
## Practical Expertise in Quantized LLMs
 
Quantization of LLMs is a detailed and nuanced process, bringing together theoretical concepts of machine learning and practical aspects of software engineering. With advanced tools like llama.cpp, deployment of these efficient models is increasingly becoming more feasible across various hardware platforms. Mastery of quantization techniques is essential for anyone looking to efficiently deploy LLMs in real-world applications.
 
This synopsis provides an overview of a deeply technical discussion on the subject of LLM quantization, highlighting the principal components.