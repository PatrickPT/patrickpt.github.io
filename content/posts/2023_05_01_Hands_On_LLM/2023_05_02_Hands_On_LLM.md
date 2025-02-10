---
title: "What are Large Language Models?"
date: 2023-05-02T13:42:28+02:00
draft: false
summary: What are the ideas and concepts behind GPT Language Models?
showToc: true
TocOpen: true
url: /posts/language_models/
tags: [LLM]
---


# TL;DR
A short summary on (Large) Language Models: What are the ideas and concepts behind Language Models?

# What are Language Models

Large Language Models are rapidly transforming natural language processing tasks and have led to a huge hype around Artificial Intelligence and associated tasks. Especially with the introduction of GPT3.5(ChatGPT) in 2022 many people are interested in the capabilities of ML.
But what is the foundation for the products that could impact our future world significantly?

Large language models rely basically on neural networks – more specifically, deep neural networks – to process natural language and can be trained on large amounts of text data, such as books, articles, and social media posts. They parse through the data and generate predictions about what words will come next in a sentence or phrase. 

The larger the dataset, the more information that can be fed into the network for analysis. This increases the performance since the model has more data to evaluate against.

One of the key techniques used in large language models is called "pretraining." This involves training the model on a large corpus of text data using a technique called unsupervised learning, where the model tries to learn the underlying structure of the language without any explicit supervision. This pretraining step helps the model to learn a general understanding of the language, which can then be fine-tuned for specific tasks like language translation or text generation.

Another important concept in large language models is "attention," which refers to the ability of the model to focus on specific parts of the input sequence when making predictions. Attention mechanisms allow the model to selectively weight the importance of different parts of the input sequence based on their relevance to the task at hand.

# Architectures

There are several architectures used for Large Language Models (LLMs), but some of the most popular ones are:

- **Recurrent Neural Networks (RNNs)** RNNs are a type of neural network that is commonly used for processing sequences, including natural language. They work by processing each input element (in this case, each word in a sentence) in a sequence, and updating their internal state based on the previous inputs. This makes them well-suited for language modeling tasks, where the model needs to keep track of the context of the sentence to make predictions.
- **Convolutional Neural Networks (CNNs)**: CNNs are another type of neural network that have been used for LLMs. They are typically used for image recognition tasks, but they can also be used for language processing tasks by treating the words in a sentence as a 1-dimensional signal. In this case, the convolution operation can help the model to identify important features of the sentence, such as n-grams or phrases.
- **Transformer Models**: Transformer models are a more recent architecture that has been widely used for LLMs. They are based on a self-attention mechanism that allows the model to selectively attend to different parts of the input sequence. This makes them highly effective for modeling long-range dependencies in text data, and they have been used to achieve state-of-the-art results in a wide range of language processing tasks.
- **GPT (Generative Pre-trained Transformer) Models**: GPT models are a type of transformer model that are pre-trained on large amounts of text data using unsupervised learning. They have been used for a wide range of natural language processing tasks, including language translation, text summarization, and text generation. The GPT series has evolved over time with each version improving on the previous in terms of size and performance.
- **BERT (Bidirectional Encoder Representations from Transformers) Models**: BERT models are also based on transformer architecture but are trained using a different pre-training objective. They are trained to understand the context of words based on their surrounding words, both before and after, unlike traditional LLMs which are only trained to consider context before the word. This makes BERT models particularly useful for tasks such as question answering and sentiment analysis.


Each of these architectures has its own strengths and weaknesses, and the choice of architecture depends on the specific task and the nature of the data being processed. However, in recent years, transformer-based models such as GPT and BERT have emerged as the most effective architectures for large language modeling tasks.

# Why the Hype

The statistical methods behind LLM's are rather simple. The recent innovation comes from investing a huge amount of money and a clever strategy to finetune. Reinforcement Learning on Human Feedback.

Reinforcement Learning with Human Feedback (RLHF) is a subfield of reinforcement learning (RL) that aims to incorporate human feedback into the training of RL agents. In RLHF, humans provide feedback to the RL agent during the training process to help guide its behavior.

The basic idea behind RLHF is to leverage the unique strengths of both humans and machines. While machines are capable of processing large amounts of data quickly and making decisions based on objective criteria, humans are better at handling uncertainty, understanding complex contexts, and incorporating ethical considerations into decision-making.

There are several approaches to incorporating human feedback into RL training. One common approach is called "reward shaping," where humans provide additional reward signals to the agent to help guide its behavior. For example, if an agent is learning to play a game, a human might provide additional rewards for certain actions that the agent is not considering on its own.

Another approach is called "active learning," where the agent actively seeks out feedback from humans to improve its performance. For example, the agent might ask a human to provide feedback on a specific decision it is considering, or it might ask for help in identifying certain features of the environment that are important for making decisions.

Overall, RLHF has the potential to make RL agents more effective in real-world applications, where the agent must interact with humans or operate in complex, uncertain environments. However, it also raises important ethical considerations, such as how to ensure that the human feedback is fair and unbiased, and how to handle situations where the human feedback is conflicting or inconsistent.

# Hands On

As said it is rather simple to set up your own Language Model and if you are eager to do it you can use the nanoGPT implementation of Andrej Karpathy on GitHub](https://github.com/karpathy/nanoGPT) to test out your own model.

# Ressources

[Large Language Models](https://huggingface.co/blog/large-language-models)

[ChatGPT](https://chat.openai.com/chat)

[nanoGPT](https://github.com/karpathy/nanoGPT)

