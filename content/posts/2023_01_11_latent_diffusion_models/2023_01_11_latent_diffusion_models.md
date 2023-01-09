---
title: "Latent Diffusion Models: What is all the fuzz about?"
date: 2023-01-16T14:50:18+01:00
draft: true
showToc: true
TocOpen: true
math: true
---

If you recently introduced yourself to strangers and told them about your job in AI, there is a good chance they asked you about the current hype-train around the next generation of generative AI models and related data products like Dall-E, Stable Diffusion or Midjourney.

Latent Diffusion Models like the ones above had some significant media attention. While no one outside AI community bats an eye if Deepmind creates an algorithm, that beats the (almost ancient) and important Strassen-Algorithm by some percent in computation complexity(which is a tremendous progress), nearly everyone is excited to create made up pictures of cats doing crazy stuff through a simple interface.

[![cat](/posts/2023_01_11_latent_diffusion_models/images/cat.png)](/posts/2023_01_11_latent_diffusion_models/images/cat.png)


*Dall-E 2 created picture by author - "A cat surfing a wave in comic style during sunset"*

While those models already made a name for themselves by winning [art competitions](https://news.artnet.com/art-world/colorado-artists-mad-ai-art-competition-2168495), are adopted by companies into their related data products(Canva.com, Shutterstock.com) and start-ups creating those products raising billions in [venture capital](https://www.bloomberg.com/news/articles/2022-10-17/digital-media-firm-stability-ai-raises-funds-at-1-billion-value) you may ask yourself:
<ul>
<li>What is all the fuzz about? </li>
<li>What is behind the hype? What are Latent Diffusion Models? </li>
<li>What is the math behind them? </li>
<li>Do they impact my life? What is the best way to leverage their power? </li> 
</ul>
Let me briefly introduce you to Latent Diffusion Models, explain the math and also give you some examples on how to use them.

# What is this post about:

A brand-new category of cutting-edge generative models called diffusion models generate a variety of high-resolution images.
They have already received a great deal of attention as a result of OpenAI, Nvidia, and Google's success in training massive models.
Examples of diffusion model-based designs include GLIDE, DALLE-2, Imagen, and the complete open-source stable diffusion.
We will try to understand what those models are and further look into the main mathematical principles behind Latent Diffsuion Models
I will give an overview and mathematical intuition on the most prominent diffusion-based architecture: Denoising Diffusion Probabilistic Models (DDPM) as initialized by Sohl-Dickstein et al and proposed by Ho. et al 2020. Also I will refer to latent representation as proposed by Rombach et al.

# What are Latent Diffusion Models?

You may ask "What is the intuition behind latent diffusion models?"
Let's break it down with a short example to make it clear:
You are a painter hired by Vatican with the task to repaint the fresco at the ceiling of [sixtinian chapel](https://de.wikipedia.org/wiki/Sixtinische_Kapelle#/media/Datei:CAPPELLA_SISTINA_Ceiling.jpg).

The requirement is to recreate the fresco with pictures of cats.
Most likely you will be overwhelmed with that task and immediately think of structuring your work into smaller chunks. You want to start with one dedicated part of the whole fresco, for example the creation of Adam. You will put some effort in it, first paint the background of that chunk of the painting. Then you may outline the structures you want to paint, then you start with one object like the arm of Adam,  and then focus on the hand. Finally you are happy with the scene, decide to finish it and tackle the next part. Still later you may change things after you decided that it fits better with the overall fresco.

(#TODO Picture creation of Adam with Cats)

Finally you are building a masterpiece out of masterpieces lasting centuries until someone thinks dogs are nicer than cats(so never).
The same idea comes with Diffusion: Decompose the image to smaller chunks and create gradually the best sample possible in many small steps. Breaking up the image sampling allows the models to correct itself over those small steps iteratively and produce a good sample.

Unfortunately nothing is cost-neutral. Like it will cost a painter like Michealangelo almost 4 years to finish the fresco in Sixtinian chapel, the iterative process makes the LDM slow at sampling(At least compared to GANs).

Focussing on the process of Denoising diffusion probabilistic models and to wrap it up: Latent Diffusion Models are a type of generative model that can be used to reconstruct an input signal from a noisy version of that signal. These models are based on the concept of diffusion, which refers to the way in which a signal spreads out and becomes more diffuse over time.

[![types_gans](/posts/2023_01_11_latent_diffusion_models/images/types_gans.png)](/posts/2023_01_11_latent_diffusion_models/images/types_gans.png)
*Overview of the different types of generative [models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/HowdoestheDiffusionprocesswork?)*

At a high level, they work by first representing the input signal as a set of latent variables, which are then transformed through a series of probabilistic transformations to produce the output signal. The transformation process is designed to smooth out the noise in the input signal and reconstruct a cleaner version of the signal.

Think of going forward and backward. Diffusion models take an input image \\(x0\\)​ and gradually add Gaussian noise to it through a series of \\(T\\) steps. This is the so-called forward process. (You may immediately think of the forward pass of neural networks, but it is unrelated). This is necessary to generate the targets for our neural network (the image after applying \\(t < T\\) noise steps). Afterwards, a neural network is trained to recover the original data by reversing the noising process. 
The ability of the model to reverse the process, makes it possible to generate new data. This is the so-called reverse diffusion process (the sampling process of a generative model).

# Forward diffusion

Diffusion models are somewhat latent variable models. Latent means in this case a hidden continuous feature space.

They are formulated using a Markov chain of \\(TT\\) steps. A Markov chain means in this case that each step only depends on the previous one. But there is no constraint to using a specific type of neural network, unlike flow-based models.

Given a data-point \\( x_0\\)​ sampled from the real data distribution \\(q(x) ( x_0 \sim q(x))\\), one can define a forward diffusion process by adding noise. Specifically, at each step of the Markov chain we add Gaussian noise with variance \\(\beta_{t}βt​ to \textbf{x}_{t-1}xt-1\\)​, producing a new latent variable \\(\textbf{x}_{t}xt\\)​ with distribution \\( q(\textbf{x}_t |\textbf{x}_{t-1})q(xt​∣xt-1​)\\). This diffusion process can be formulated as follows:
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_t=\sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \boldsymbol{\Sigma}_t = \beta_t\mathbf{I})q(xt​∣xt-1​)=N(xt​;μt​=1-βt​​xt-1​,Σt​=βt​I)
$$