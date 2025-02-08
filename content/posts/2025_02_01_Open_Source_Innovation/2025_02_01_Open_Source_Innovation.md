---
title: "Why Open Source will benefit from DeepSeek"
date: 2025-02-02T09:30:57Z
draft: true
ShowToc: true
tags: [LLM, GenAI, Architecture]
math: true
url: /posts/deepseek/

---

# TL;DR

This blog post explains how the recent release of DeepSeek may benefit the open source community and why it is considered a game changer for AI industry.

---

# Why all the rumors?

DeepSeek-R1 represents a significant innovation in the AI landscape, outperforming or rivaling top commercial models including reasoning capabilities. Previously, such sophisticated models were exclusive to tech giants like OpenAI and Google, but R1 now joins this category as the only open-weights model of its kind. Its Open Source approach including a MIT license further amplifies its disruptive potential, enabling unrestricted commercial use, even for direct competitors, without costly R&D.

Beyond accessibility, DeepSeek-R1 heats innovation by openly sharing its cost-effective training methodology, challenging the narrative that advanced AI requires exorbitant GPU investments. Operationally, it’s remarkably affordable: output tokens cost 30x less than OpenAI’s, and input tokens are 55x cheaper. 

R1’s blend of performance, openness, innovation and affordability could signal that open-source can still play a pivotal role in the AI race.

---
# Key Facts at a glance


## **Key Training Process & Innovations in DeepSeek-R1**  

### **1. Two Core Models**  
- **DeepSeek-R1-Zero**:  
  - **Pure RL Training**: Trained *directly* on the base model (DeepSeek-V3-Base) **without supervised fine-tuning (SFT)**.  
  - **Algorithm**: Uses **GRPO** (Group Relative Policy Optimization), a critic-free RL method that estimates baselines from group scores.  
  - **Reward Design**: Relies on **rule-based rewards** (accuracy + formatting) instead of neural reward models to avoid reward hacking.  
  - **Emergent Behaviors**: Self-verification, reflection, and extended reasoning (e.g., "aha moments") emerged naturally during RL.  
  - **Limitations**: Poor readability, language mixing, and formatting inconsistencies.  

- **DeepSeek-R1**:  
  - **Cold-Start Data**: Starts with **thousands of curated CoT examples** to fine-tune the base model, improving readability and stability.  
  - **Multi-Stage Training**:  
    1. **Cold-Start SFT**: Initial fine-tuning for readable CoT generation.  
    2. **Reasoning-Oriented RL**: Applies GRPO with added **language consistency rewards**.  
    3. **Rejection Sampling + SFT**: Generates high-quality data (600k reasoning + 200k non-reasoning samples) for retraining.  
    4. **Final RL Alignment**: Balances reasoning performance with human preferences (helpfulness/harmlessness).  


### **2. Key Innovations**  
1. **RL-First Approach**:  
   - Proves reasoning capabilities can be **incentivized purely through RL** without SFT (novel for open research).  
   - Achieves OpenAI-o1-0912-level performance (e.g., 71% → 86.7% on AIME with majority voting).  

2. **Cold-Start Strategy**:  
   - Addresses R1-Zero’s limitations by seeding RL with human-prioritized data (readability, structured outputs).  

3. **Distillation Efficiency**:  
   - Smaller models (1.5B–70B) distilled from DeepSeek-R1 outperform RL-trained counterparts (e.g., 72.6% vs. 47% on AIME for Qwen-32B).  
   - Distilled models surpass GPT-4o/Claude-3.5-Sonnet in math/coding tasks despite smaller size.  

4. **Rule-Based Rewards**:  
   - Avoids neural reward models, simplifying training and reducing hacking risks.  


### **3. Critical Challenges & Insights**  
- **Failed Attempts**:  
  - **Process Reward Models (PRMs)**: Struggled with fine-grained step validation and scalability.  
  - **Monte Carlo Tree Search (MCTS)**: Token-generation complexity made iterative improvement impractical.  
- **Key Insight**: Distillation is more cost-effective than RL for smaller models, but advancing SOTA requires large-scale RL on powerful base models.  


### **4. Performance Highlights**  
- **DeepSeek-R1**: Matches **OpenAI-o1-1217** on reasoning (79.8% pass@1 on AIME) and outperforms GPT-4o/Claude-3.5 in math/coding.  
- **Distilled Models**:  
  - 7B model surpasses GPT-4o on MATH-500 (92.8% vs. 74.6%).  
  - 32B model outperforms QwQ-32B-Preview by 22.6% on AIME.  


**Why It Stands Out**:  
- First open-source work validating pure RL for reasoning.  
- Combines scalability (GRPO), human-aligned cold-start data, and efficient distillation.  
- Open-sources models/data, enabling community-driven advancements.


# Understanding what DeepSeek did

To gain a clearer insight into the core framework of DeepSeek-R1, let’s break down its foundational concepts:

**Reinforcement Learning (RL):** This approach involves a model learning through a system of rewards and penalties tied to its actions, refining its performance over time via trial and error. In the realm of large language models (LLMs), RL can be implemented through techniques such as policy optimization (e.g., Proximal Policy Optimization or PPO), value-based methods (e.g., Q-learning), or combined approaches like actor-critic architectures. For instance, when presented with a prompt like “2 + 2 =”, the model might receive a reward of +1 for generating the correct answer “4” and a penalty of -1 for any incorrect response. In advanced LLMs, rewards are often derived from human feedback (RLHF) or automated evaluation systems like GRPO.

**Supervised Fine-Tuning (SFT):** This process involves retraining a base model using a labeled dataset to enhance its performance on a specific task. For example, an LLM could be fine-tuned with a dataset of customer service queries and responses to improve its accuracy in addressing common support questions. This method is particularly effective when a substantial amount of labeled data is available.

**Cold Start Data:** This refers to a small, minimally labeled dataset used to provide the model with a basic grasp of the task at hand. For instance, a chatbot might be fine-tuned using a simple dataset of frequently asked questions (FAQs) extracted from a website, helping it establish a foundational understanding. This approach is especially useful when labeled data is scarce.

**Multi-Stage Training:** In this method, the model undergoes training in distinct phases, each targeting a specific improvement, such as accuracy or alignment with user expectations. For example, a model might first be trained on general text data and then further refined using reinforcement learning based on user feedback to enhance its conversational capabilities.

**Rejection Sampling:** This technique involves generating multiple potential outputs from a model, but only retaining those that meet predefined criteria, such as quality or relevance. For example, after a reinforcement learning process, the model might produce several responses, but only the most useful ones are selected for retraining or further use. This ensures that only high-quality outputs contribute to the model’s ongoing improvement.


---

# How Does DeepSeek work?

---

# Implications


---


---

# Pitfalls


---

# Conclusion


# Ressources

https://thelmbook.com/articles/#!./DeepSeek-R1.md

https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

https://www.vellum.ai/blog/the-training-of-deepseek-r1-and-ways-to-use-it