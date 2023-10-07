---
title: "AI Architecture: What is Retrieval Augmented Generation"
date: 2023-09-29T18:30:57Z
draft: False
ShowToc: true
tags: [Model Architecture,AI Design, LLM, GenAI]
math: true
url: /posts/rag/
---

# TL;DR

This blogpost focusses on AI Architecture and specifically on Retrieval Augemnted Generation. Retrieval Augmented Generation can be used to productionize LLM models for enterprise architecture easily.

# Why should i care?

Intuitive would be to train a Large Language Model with domain specific data, in other words, to fine-tune the model-weights with custom data. But fine-tuning large language models (LLMs) is a complex and resource-intensive process due to several key factors:
- Acquiring the massive computational infrastructure required to train LLMs effectively demands significant financial investment. Even if you rely on Cloud Providers instead of on-premise the cost to train is a factor which cannot be neglected currently.
- Assembling high-quality, domain-specific datasets for fine-tuning can be time-consuming and expensive, as it often involves labor-intensive data collection and annotation efforts. 

Other downsides are:
- **Hallucinations:** If you think about it consequently fine-tuning will not guarantee that the model behaves the way you would like it to behave. We are able to steer how the model is "learning" when fine-tuning but we cannot control the outcome. The probability that we receive hallucinations is way higher if we fine-tune in contrast to if we just directly provide additional context in the prompt which is matching to our question.
- **Retraining and Knowledge Cut-offs:** It would require constant compute-intensive retraining for even small changes. As you would not do that every day every LLM has a training end date, post which it is unaware of events or information.
- **Context Window:** Each Large Language Model (LLM) functions within a contextual window, which essentially defines the maximum volume of information it can simultaneously accommodate. When external data sources provides information surpassing this window's capacity, it needs to be segmented into smaller portions that align with the model's contextual limitations.

# Use Retrieval Augmented Generation

Research on NLP was there before the hype around GPT's and specifically on OpenAIs service ChatGPT(based on GPT3.5) started in 2022.
In 2020 Lewis et al. from Meta AI published a paper on [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

Retrieval-augmented generation (RAG) is an AI architecture for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLMs internal representation of information.

*“It’s the difference between an open-book and a closed-book exam,”* Lastras said. *“In a RAG system, you are asking the model to respond to a question by browsing through the content in a book, as opposed to trying to remember facts from memory.”*

It is a fusion of two powerful NLP techniques: retrieval-based models and generative models. Retrieval-based models excel at finding relevant information from a vast pool of knowledge, often in the form of pre-existing text or documents. On the other hand, generative models are proficient in generating human-like text based on the given input.

Incorporating it into an LLM architecture enables the model to combine the best of both worlds. 
- It can retrieve contextually relevant information from a vast knowledge base and 
- use that information to generate coherent and contextually appropriate responses. 

This approach leads to more accurate and context-aware interactions with the language model.

![](/posts/2023_09_29_Retrieval_Augmented_Generation/images/rag_high_level.jpg)
*picture from [llama_index](https://gpt-index.readthedocs.io/en/latest/getting_started/concepts.html)*


The architecture of an LLM incorporating retrieval augmented generation typically consists of two main components:

# Retrieval Module
The retrieval module is the heart of the architecture and responsible for searching and retrieving relevant information from a large knowledge base. This knowledge base can be a collection of texts, documents, or even web pages. Retrieval can be broken down into two stages:

## Indexing: 
You will index the knowledge base which you want to give as context to the model. To index your data you can simply use libraries like LangChain llama_index or transformers which do the work for you. The data is loaded, chunked and tokenized. Data will be fetched from your various sources, segmented into bite-sized chunks to optimize them for embedding and search and afterwards tokenized to create embeddings.
The embeddings are stored as high dimensional vectors in vector databases and build the foundation for RAG.

![](/posts/2023_09_29_Retrieval_Augmented_Generation/images/indexing.jpg)
*picture from [llama_index](https://gpt-index.readthedocs.io/en/latest/getting_started/concepts.html)*

## Querying:
The data from the vector databases can be used for Semantic Search. Meaning that when inputing a question into the retriever(which was also transformed into an embedding) we can search for the most appropriate matching data in the vector database and give it to generator as context. The retrieved data is combined with the original prompt, creating an enhanced or augmented prompt. This augmented prompt provides additional context. This is even more relevant for domain specific data which may not be part of the corpus used for training the model.

![](/posts/2023_09_29_Retrieval_Augmented_Generation/images/querying.jpg)
*picture from [llama_index](https://gpt-index.readthedocs.io/en/latest/getting_started/concepts.html)*

In summary: 
We start by searching for relevant documents or excerpts within an extensive dataset. For this a dense retrieval mechanism that leverages embeddings to represent both the query and the documents is used. These embeddings are then utilized to calculate similarity scores, leading to the retrieval of the top-ranked documents.

# Generation Module
The generation module is based on a generative language model like GPT-3.5. It takes as input the retrieved information and the user's query or context and generates a coherent response.

# Example

An example architecture using llama_index would look like the following:

![](/posts/2023_09_29_Retrieval_Augmented_Generation/images/querying.jpg)

The code for this example can be found on [github](https://github.com/PatrickPT/RAG_LLM_example).
Also you can find a dedicated blogpost on my blog [Hands on with Retrieval Augmented Generation?](/posts/hands-on-rag).

# Benefits of using RAG

**Improved Precision:**
Through the utilization of external data sources, the RAG LLM can produce responses that are not solely derived from its training dataset but are also enriched by the most pertinent and current information accessible within the retrieval corpus.

**Addressing Knowledge Gaps:**
RAG effectively confronts the inherent knowledge limitations of LLM, whether stemming from the model's training cut-off or the absence of domain-specific data in its training corpus.

**Adaptability:**
RAG can seamlessly integrate with an array of external data sources, encompassing proprietary databases within organizations and publicly accessible internet data. This adaptability makes it suitable for a broad spectrum of applications and industries.

**Mitigating Hallucinations:**
A challenge associated with LLMs is the potential occurrence of "hallucinations,". By incorporating real-time data context, RAG decreases the probability of such outputs.

**Scalability:**
An advantage of RAG LLMs lies in its scalability. Through the separation of the retrieval and generation processes, the model can manage extensive datasets, making it well-suited for real-world scenarios characterized by abundant data.

# Conclusion

Retrieval augmented generation is a versatile Arhcitecture for LLM's. 

It combines the strengths of retrieval-based models and generative models to create contextually aware, intelligent, and flexible AI systems. With the ability to retrieve information from vast knowledge bases and generate human-like text, this architecture has the potential to be used for enteprises and flexible solutions.


# Ressources

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

[IBM: What is retrieval-augmented generation?](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

[Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)

[Retrieval meets Long Context Large Language Models](https://arxiv.org/abs/2310.03025)

[A Deep Dive into Retrieval-Augmented Generation in LLM](https://www.unite.ai/a-deep-dive-into-retrieval-augmented-generation-in-llm/)