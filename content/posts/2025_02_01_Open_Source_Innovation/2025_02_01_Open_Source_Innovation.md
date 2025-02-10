---
title: "DeepSeek-R1 explored"
date: 2025-02-09T18:30:57Z
draft: false
ShowToc: true
summary: Why DeepSeek-R1 was a significant step for Open Source AI.
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
# In s Nutshell


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

---

# Understanding what DeepSeek did in more detail

To gain a clearer insight into the core framework of DeepSeek-R1, let’s break down its foundational concepts:

**Reinforcement Learning (RL):** This approach involves a model learning through a system of rewards and penalties tied to its actions, refining its performance over time via trial and error. In the realm of large language models (LLMs), RL can be implemented through techniques such as policy optimization (e.g., Proximal Policy Optimization or PPO), value-based methods (e.g., Q-learning), or combined approaches like actor-critic architectures. For instance, when presented with a prompt like “2 + 2 =”, the model might receive a reward of +1 for generating the correct answer “4” and a penalty of -1 for any incorrect response. In advanced LLMs, rewards are often derived from human feedback (RLHF) or automated evaluation systems like GRPO.

**Supervised Fine-Tuning (SFT):** This process involves retraining a base model using a labeled dataset to enhance its performance on a specific task. For example, an LLM could be fine-tuned with a dataset of customer service queries and responses to improve its accuracy in addressing common support questions. This method is particularly effective when a substantial amount of labeled data is available.

**Cold Start Data:** This refers to a small, minimally labeled dataset used to provide the model with a basic grasp of the task at hand. For instance, a chatbot might be fine-tuned using a simple dataset of frequently asked questions (FAQs) extracted from a website, helping it establish a foundational understanding. This approach is especially useful when labeled data is scarce.

**Multi-Stage Training:** In this method, the model undergoes training in distinct phases, each targeting a specific improvement, such as accuracy or alignment with user expectations. For example, a model might first be trained on general text data and then further refined using reinforcement learning based on user feedback to enhance its conversational capabilities.

**Rejection Sampling:** This technique involves generating multiple potential outputs from a model, but only retaining those that meet predefined criteria, such as quality or relevance. For example, after a reinforcement learning process, the model might produce several responses, but only the most useful ones are selected for retraining or further use. This ensures that only high-quality outputs contribute to the model’s ongoing improvement.

---

## How Does DeepSeek Work?

DeepSeek-R1 is built on a multi-stage training regime that combines a carefully engineered supervised fine-tuning (SFT) phase with pure reinforcement learning (RL) techniques—most notably, a novel Group Relative Policy Optimization (GRPO) framework. This section outlines the architecture, training pipeline, and mathematical formulation underpinning DeepSeek-R1.

### 1. Architectural Overview

At its core, DeepSeek-R1 is based on a powerful base model (DeepSeek-V3-Base) which is subsequently refined through several stages:

- **Base Model Initialization:**  
  The process begins with a pre-trained large language model. This base model, having been trained on vast internet-scale data, provides the foundational language understanding and generative capabilities.

- **Chain-of-Thought (CoT) Representation:**  
  A unique aspect of DeepSeek-R1 is its explicit generation of chain-of-thought reasoning. During inference, the model produces both the reasoning steps (CoT) and the final answer. This transparency is achieved by encouraging multi-step reasoning during training.

### 2. Multi-Stage Training Pipeline

DeepSeek-R1’s training is divided into several well-defined phases:

#### a. Cold-Start Supervised Fine-Tuning (SFT)

- **Objective:**  
  To provide the model with an initial “readable” structure and coherent reasoning behavior.  
- **Method:**  
  A relatively small but high-quality dataset of thousands of chain-of-thought examples (referred to as *cold-start data*) is used to fine-tune the base model. Although the dataset is orders of magnitude smaller than typical supervised corpora, its curation for clarity and structured output is crucial for overcoming issues like language mixing and formatting inconsistencies.

#### b. Pure Reinforcement Learning with GRPO

- **Objective:**  
  To enhance reasoning capabilities by learning directly from trial-and-error without any further labeled data.  

- **GRPO Overview:**  
  Traditional RL approaches in language modeling often employ a critic network or neural reward models. Instead, DeepSeek-R1 uses GRPO—a critic-free method where rewards are determined via rule-based measures (e.g., coherence, formatting, and logical consistency) and then compared relative to group-average scores.  

- **Mathematical Formulation:**  

  Let the policy of the model be denoted as $\pi_\theta(y|x)$ (with parameters $\theta$), and assume a reference policy $\pi_{\text{ref}}(y|x)$ derived from the cold-start SFT stage. For a given prompt $x$ and two candidate responses $y_w$ (winning) and $y_l$ (losing), define the relative log-likelihood difference as:
  
  $$
  h_{\pi_\theta}(x, y_w, y_l) = \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
  $$
  
  The standard Direct Preference Optimization (DPO) loss (as used in earlier works) is given by:
   
   The DPO loss is computed as the expected value (over the data distribution $\(D\))$ of the negative log-sigmoid of a scaled difference $\(h_{\pi_\theta}\)$ between a winning and a losing candidate.
   This means we are averaging the penalized log probability over the sample pairs.
  
  where $\sigma(\cdot)$ is the sigmoid function and $\beta$ is a scaling factor.  

  In GRPO, to account for multiple groups $g \in \{1, \ldots, K\}$ (which may correspond to different task domains or user preferences), the objective is reformulated as a worst-case (minimax) problem:
  
  $$
  L_{\text{GR}}(\pi_\theta) = \max_{g \in \{1,\ldots,K\}} \; L_{\text{DPO}}(\pi_\theta; D_g)
  $$
  
  Alternatively, using a weighted combination over groups with weights $\alpha \in \Delta_K$ (the $K$-simplex), we have:
  
 We are optimizing the model by minimizing over the policy \( \pi_\theta \) while considering the worst-case (or maximum over groups) loss. For each group \( g \), the loss is computed as the expected negative log-sigmoid of the scaled difference \( h_{\pi_\theta} \) between winning and losing responses. In other words, the model focuses on improving the worst-performing group by weighting the loss accordingly.
  
  This formulation ensures that groups with poorer performance (i.e., higher loss) receive higher weights during training, guiding the model to improve in those areas.

- **Gradient Update:**  
  The gradient update for the parameters $\theta$ is derived from the weighted loss. If we denote the loss on a sample as:
  
  $$
  l(\pi_\theta; (x_g, y_w, y_l)) = \log \sigma\left(\beta \cdot h_{\pi_\theta}(x_g, y_w, y_l)\right)
  $$
  
  then the gradient update (ignoring normalization factors) is:
  
  $$
  \nabla_\theta l(\pi_\theta; (x_g, y_w, y_l)) \propto \sigma\Big(r_\theta(x_g, y_l) - r_\theta(x_g, y_w)\Big) \left[ \nabla_\theta \log \pi_\theta(y_w|x_g) - \nabla_\theta \log \pi_\theta(y_l|x_g) \right]
  $$
  
  where $r_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$. This update explicitly increases the probability of the preferred response while decreasing that of the rejected one, with additional weighting from the group-specific factors $\alpha_g$.


#### c. Rejection Sampling and Synthetic Data Generation

- **Objective:**  
  To refine the model’s outputs by selecting only high-quality samples.
- **Method:**  
  Once the RL process has nearly converged, the model generates multiple outputs for a given prompt. A rejection sampling mechanism filters these outputs using the same rule-based criteria (coherence, formatting, etc.), creating a synthetic dataset of high-quality reasoning examples. This dataset typically contains on the order of 600k reasoning samples and 200k non-reasoning samples, which are then used to further fine-tune the model via supervised training.

#### d. Final RL Alignment

- **Objective:**  
  To balance high-level reasoning with human-aligned attributes such as helpfulness and harmlessness.
- **Method:**  
  A final round of RL is applied over diverse prompts. This stage consolidates improvements from the previous stages while ensuring that the model’s behavior aligns with user expectations. The final objective remains similar to the GRPO loss, ensuring that no particular subgroup (or aspect of the task) is neglected.

### 3. Inference: Chain-of-Thought Generation

During inference, DeepSeek-R1 generates a two-part output:
- **Reasoning Content:**  
  The intermediate chain-of-thought (CoT) that details the step-by-step reasoning process.
- **Final Answer:**  
  The concise output generated after the reasoning steps.

The dual output is a direct consequence of the multi-stage training pipeline. The training not only instills raw reasoning power but also structures the output to separate the thought process from the final answer—an innovation that facilitates transparency and error analysis.

### 4. Distillation into Smaller Models

An additional contribution of DeepSeek-R1 is the distillation of its reasoning capabilities into smaller models (ranging from 1.5B to 70B parameters). The distilled models inherit the chain-of-thought behavior and high reasoning performance, often outperforming models trained solely with RL. This distillation leverages the observation that reasoning patterns discovered by larger models can be effectively transferred to compact architectures without sacrificing performance.

---

## Summary

DeepSeek-R1 works through a meticulously designed multi-stage training process:
1. **Cold-Start SFT** to provide a solid, human-readable foundation.
2. **Pure RL using GRPO** to boost reasoning capability by optimizing for worst-case group performance.
3. **Rejection Sampling and SFT** to refine outputs by creating a high-quality synthetic dataset.
4. **Final RL Alignment** to ensure the model adheres to human values and achieves a balanced performance.

Mathematically, the model’s optimization is expressed via a robust version of the Direct Preference Optimization (DPO) loss, where group-specific losses are balanced through a minimax formulation.

---

# Implications

The release of DeepSeek-R1 was not just another model but it represents a shift in how advanced LLMs can be built and deployed:

## Democratization of Advanced AI:
Against the overwhelming trend of closed-source models, DeepSeek-R1 is open-sourced and still reaches on-par performance with commercial models. This openness invites collaborative improvements and accelerates innovation in AI.

## Cost-Effective Scaling:
DeepSeek-R1 dramatically reduces token costs (30× cheaper for outputs and 55× cheaper for inputs than comparable commercial models). This cost efficiency challenges the narrative that state-of-the-art reasoning demands vast GPU clusters and massive labeled datasets, initiating new innovation in AI training paradigms.

## Validation of Pure RL for Reasoning:
The success of DeepSeek-R1-Zero proves that advanced reasoning can emerge solely through reinforcement learning. This breakthrough inspires further research into RL-first training paradigms, potentially reshaping how future LLMs are developed.

## Influence on Commercial Strategies:
By openly publishing its training methods—including its efficient distillation and group-based optimization techniques—DeepSeek-R1 pressures commercial entities to adopt more transparent and community-friendly approaches. This could lead to a more competitive and innovative AI ecosystem.

---

# Pitfalls
While DeepSeek-R1 sets a new standard, several challenges remain:

- **Readability and Formatting:**  
  The initial R1-Zero model exhibited issues such as poor readability and language mixing. Although the multi-stage training process addresses these concerns, some residual inconsistencies may persist in complex scenarios.

- **Inference Latency:**  
  DeepSeek-R1’s chain-of-thought generation is computationally intensive. Inference times are slower compared to models optimized solely for speed, which may limit its use in latency-critical applications.

- **Limited Inference Flexibility:**  
  The current API version does not support many adjustable parameters (e.g., temperature, top_p, etc.), making it harder to fine-tune output behavior for production environments.

- **Risk of Overfitting to Rule-Based Rewards:**  
  Relying on fixed, rule-based rewards simplifies training but may also constrain the model’s adaptability. In cases where nuanced human judgment is required, this approach might not capture every subtlety.

- **Scalability of Pure RL:**  
  Although pure RL has proven effective here, it typically requires longer training times and can be sensitive to reward design. The balance between cost-effectiveness and training complexity remains a delicate one.

---

# Conclusion

In summary, DeepSeek-R1 shows that it’s possible to achieve advanced reasoning capabilities in large language models without the traditionally high costs. By using a modest yet carefully curated cold-start supervised phase combined with pure reinforcement learning via GRPO—and further refined with rejection sampling—the model delivers performance that can stand alongside commercial systems like those from OpenAI. Its open-source approach and transparent training process offer practical benefits for both researchers and practitioners.

That said, DeepSeek-R1 is not without its challenges. There remain issues with output readability, inference speed, and some limitations in fine-tuning flexibility. These factors indicate that there is still room for improvement as we continue to explore and refine these methods.

Overall, DeepSeek-R1 represents a thoughtful step forward in the development of cost-efficient, robust, and accessible AI. As the community builds on these ideas, we can expect further progress that will help broaden the range of available tools and techniques in the field.

---

# Ressources

https://thelmbook.com/articles/#!./DeepSeek-R1.md

https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

https://www.vellum.ai/blog/the-training-of-deepseek-r1-and-ways-to-use-it

https://arxiv.org/pdf/2405.20304