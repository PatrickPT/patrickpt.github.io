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

This blog post explains how the recent archictectural innovations from DeepSeek may benefit the open source community and why they are considered a game changer for AI industry.

---

# Why Should I Care?

1. It beats or matches the leading commercial LLMs. It's especially shocking that it beats OpenAI's o1, the model that "thinks" before answering.

2. R1 is also a thinking model, which previously only OpenAI and Google could develop (and only OpenAI's model was good at it).

3. Given the above two points, R1 is the only such model with open weights.

4. Not just open weights, but MIT licensed, so anyone can use it for any purpose, build a business directly competing with OpenAI and other big players without spending a dollar on R&D.

5. DeepSeek released the secret sauce of how to train such a model from scratch, so OpenAI and Google don't have any competitive advantage going forward. And this secret sauce is very cheap to implement, compared to what big players made us believe through their hyped GPU spending.

6. R1 is super cheap to run too. For example, OpenAI charges $60.00 per 1M output tokens for o1, while R1 only costs $2. For input tokens: OpenAI charges $15.00 per 1M tokens, while R1 will charge you $0.27.

So, combining all these factors, it's a market disruption unseen before. Hence the sad smiles from Satya and Sam. They still have money and GPUs, but they don't know what to do with it anymore.   

---

# Understanding what DeepSeek did

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

---

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

---

### **3. Critical Challenges & Insights**  
- **Failed Attempts**:  
  - **Process Reward Models (PRMs)**: Struggled with fine-grained step validation and scalability.  
  - **Monte Carlo Tree Search (MCTS)**: Token-generation complexity made iterative improvement impractical.  
- **Key Insight**: Distillation is more cost-effective than RL for smaller models, but advancing SOTA requires large-scale RL on powerful base models.  

---

### **4. Performance Highlights**  
- **DeepSeek-R1**: Matches **OpenAI-o1-1217** on reasoning (79.8% pass@1 on AIME) and outperforms GPT-4o/Claude-3.5 in math/coding.  
- **Distilled Models**:  
  - 7B model surpasses GPT-4o on MATH-500 (92.8% vs. 74.6%).  
  - 32B model outperforms QwQ-32B-Preview by 22.6% on AIME.  

--- 

**Why It Stands Out**:  
- First open-source work validating pure RL for reasoning.  
- Combines scalability (GRPO), human-aligned cold-start data, and efficient distillation.  
- Open-sources models/data, enabling community-driven advancements.

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

