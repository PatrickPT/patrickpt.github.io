---
title: "Latent Diffusion Models: What is all the fuzz about?"
date: 2023-01-20T14:50:18+01:00
draft: false
summary: Learn how Latent Diffusion Models work and the mathematical intuition behind them.
showToc: true
TocOpen: true
math: true
tags: [Math,stable-diffusion,latent-diffusion-models,Notes]
url: /posts/latent-diffusion-models/

---

# TL;DR
The following learning notes try to give some intuition on how [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) works, some mathematical intuition on Diffusion Models and an introduction to Latent Diffusion Models.

**The following sources were extensively used during creation of this learning notes. Passages may be reutilized to create a reasonable overview of the topic. Credit goes to the authors of the following papers and posts.**

[Illustrations by Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/)

[Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)

[Ho et al. 2020](https://arxiv.org/abs/2006.11239)

[Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752)

[Lilian Weng on Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

[Sergios Karagiannakos on Diffusion Models](https://theaisummer.com/diffusion-models/)

[Hugging Face on Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

If you recently introduced yourself to strangers and told them about your job in AI, there is a good chance they asked you about the current hype-train around the next generation of generative AI models and related data products like Dall-E, Stable Diffusion or Midjourney.

Latent Diffusion Models like the ones above had some significant media attention. While no one outside AI community bats an eye if Deepmind creates an algorithm, that beats the (almost ancient) and important Strassen-Algorithm by some percent in computation complexity(which is a tremendous progress), nearly everyone is excited to create made up pictures of cats doing crazy stuff through a simple interface.

[![cat-surfing](/posts/2023_01_11_latent_diffusion_models/images/cat_surfing.jpeg)](/posts/2023_01_11_latent_diffusion_models/images/cat_surfing.jpeg)
*Stable Diffusion "cat surfing waves at sunset, comic style"*

While those models already made a name for themselves by winning [art competitions](https://news.artnet.com/art-world/colorado-artists-mad-ai-art-competition-2168495), are adopted by companies into their related data products(Canva.com, Shutterstock.com) and start-ups creating those products raising billions in [venture capital](https://www.bloomberg.com/news/articles/2022-10-17/digital-media-firm-stability-ai-raises-funds-at-1-billion-value) you may ask yourself:

* What is all the fuzz about? 
* What is behind the hype? What are Latent Diffusion Models? 
* What is the math behind them? 
* Do they impact my life? What is the best way to leverage their power?  

Let me briefly introduce you to Diffusion Models and Latent Diffusion Models and explain the math.

If you are interested in a Hands-On you can find that in my other post:

[Hands on Latent Diffusion Models](/posts/hands-on-latent-diffusion-models)

If you want an awesome visual introduction with diagrams i strongly advise to visit the [blog by Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/).

# What is this post about:

A brand-new category of cutting-edge generative models called diffusion models generate a variety of high-resolution images.
They have already received a great deal of attention as a result of OpenAI, Nvidia, and Google's success in training massive models.
We'll take a deeper look into **Denoising Diffusion Probabilistic Models** (also known as DDPMs, diffusion models, score-based generative models or simply [autoencoders](https://benanne.github.io/2022/01/31/diffusion.html)) as researchers have been able to achieve remarkable results with them for (un)conditional image/audio/video generation. Popular examples (at the time of writing) include [GLIDE](https://arxiv.org/abs/2112.10741) and [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI, [Latent Diffusion](https://github.com/CompVis/latent-diffusion) by the University of Heidelberg and [ImageGen](https://imagen.research.google/) by Google Brain.


# Intuition on Diffusion Models

You may ask "What is the intuition behind diffusion models?"
Let's break it down with a short example to make it clear:
You are a painter hired by Vatican with the task to repaint the fresco at the ceiling of [sixtinian chapel](https://de.wikipedia.org/wiki/Sixtinische_Kapelle#/media/Datei:CAPPELLA_SISTINA_Ceiling.jpg).

The requirement is to recreate the fresco with pictures of cats. The requirement is based on the old fresco and the vatican wants to have the same scenes as currently there but witch cats.
So you start remembering the fresco and start bringing up a base coat and the old fresco soon becomes a big grey noise.

Most likely you will be overwhelmed with creating a big fresco and immediately think of structuring your work into smaller chunks.
Then you may outline the structures you want to paint based on your memory on the old picture and your vision on the new one.

 Maybe you start with one object like the arm of Adam, from [the creation of Adam](https://en.wikipedia.org/wiki/The_Creation_of_Adam) and then focus on the hand. Gradually you add more details and finally are happy with the scene, decide to finish it and tackle the next part. Still later you may change things after you decided that it fits better with the overall fresco.

Finally you are building a masterpiece lasting centuries until someone thinks dogs are nicer than cats(so never).

The same idea comes with Diffusion: Gradually add noise and create the best representation of the input vision in many small steps. Breaking up the image sampling allows the models to correct itself over those small steps iteratively and produces a good sample.

Unfortunately nothing is cost-neutral. Like it will cost a painter like Michealangelo almost 4 years to finish the fresco in Sixtinian chapel, the iterative process makes the Models slow at sampling(At least compared to GANs).

# What are Diffusion Models?

The idea of diffusion for generative modeling was introduced in ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)). However, it took until ([Song et al., 2019](https://arxiv.org/abs/1907.05600)) (at Stanford University), and then ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) (at Google Brain) who independently improved the approach.
DDPM  which we are focussing on originally was introduced in a paper by ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)).

At a high level, they work by first representing the input signal as a set of latent variables, which are then transformed through a series of probabilistic transformations to produce the output signal. The transformation process is designed to smooth out the noise in the input signal and reconstruct a cleaner version of the signal.

Transformation consist of 2 processes. 

In a bit more detail for images, the set-up consists of 2 processes:
* a fixed (or predefined) forward diffusion process \\(q\\) of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise
* a learned reverse denoising diffusion process \\(p_\theta\\), where a neural network is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.

Diffusion Models are basically generative models:
[![types_gans](/posts/2023_01_11_latent_diffusion_models/images/types_gans.png)](/posts/2023_01_11_latent_diffusion_models/images/types_gans.png)
*Overview of the different types of generative [models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/HowdoestheDiffusionprocesswork?)*


# I want to see the math

## Diffusion

<!-- At a high level, Diffusion Models work by first representing the input signal as a set of latent variables, which are then transformed through a series of probabilistic transformations to produce the output signal. The transformation process is designed to smooth out the noise in the input signal and reconstruct a cleaner version of the signal. --> 

In probability theory and statistics, diffusion processes are a class of continuous-time [Markov](https://en.wikipedia.org/wiki/Markov_chain) process with almost surely continuous sample paths. E.g. [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion)

Wikipedia says:

* A Markov chain or Markov process is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event.

* A continuous-time Markov chain (CTMC) is a continuous stochastic process in which, for each state, the process will change state according to an exponential random variable and then move to a different state as specified by the probabilities of a stochastic matrix



Diffusion consists of 2 processes:
* a fixed (or predefined) forward diffusion process $q$ of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise
* a learned reverse denoising diffusion process $p_\theta$, where a neural network is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.


![DDPM](/posts/2023_01_11_latent_diffusion_models/images/DDPM_pres.png)
*The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source: Ho et al. 2020) and [Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)*

## Forward Diffusion

Given a data point from a real data distribution $x\_0 \sim q(x)$ we define a 
**forward diffusion** in which we add small Gaussian noise stepwise for $T$ steps
producing noisy samples $\mathbf{x}\_1, \dots, \mathbf{x}\_T$

Step sizes are controlled by a variance schedule $0 < \beta\_1 < \beta\_2 < ... < \beta\_T < 1$.

It is defined as

$q(x\_t | x\_{t-1}) = \mathcal{N}(x\_t; \sqrt{1 - \beta\_t} x\_{t-1}, \beta\_t \mathbf{I})$
with
$\sqrt{1 - \beta\_t} x\_{t-1}$ as decay towards origin and

$\beta\_t \mathbf{I}$ as the addition of small noise.

Rephrased:
A normal distribution (also called Gaussian distribution) is defined by 2 parameters: 
* a mean $\mu$ and 
* a variance $\sigma^2 \geq 0$. 

Basically, each new (slightly noisier) image at time step $t$ is drawn from a **conditional Gaussian distribution** with

* $\mathbf{\mu}\_t = \sqrt{1 - \beta\_t} \mathbf{x}\_{t-1}$ and 
* $\sigma^2\_t = \beta\_t$, 

which we can do by sampling $\mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ and then setting $\mathbf{x}\_t = \sqrt{1 - \beta\_t} \mathbf{x}\_{t-1} +  \sqrt{\beta\_t} \mathbf{\epsilon}$. 


Given a sufficiently large $T$ and a well behaved schedule for adding noise at each time step, you end up with what is called an [isotropic Gaussian distribution](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) at $t=T$ via a gradual process.

Isotropic means the probability density is equal (iso) in every direction (tropic). In gaussians this can be achieved with a $\sigma^2 I$ covariance matrix.

One property of the diffusion process is, that you can sample $x\_t$ at any time step $t$ using [reparameterization trick](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick).
Let $\alpha\_{t} = 1 - \beta\_t$ and $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$


$$
\begin{aligned}
\mathbf{x}_t
&= \sqrt{\alpha_t}\mathbf{x}\_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}\_{t-1} & \text{ ;where } \boldsymbol{\epsilon}\_{t-1}, \boldsymbol{\epsilon}\_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\\
&= \sqrt{\alpha_t}\mathbf{x}\_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}\_{t-1} & \text{ ;where } \boldsymbol{\epsilon}\_{t-1}, \boldsymbol{\epsilon}\_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\\
&= \sqrt{\alpha_t \alpha\_{t-1}} \mathbf{x}\_{t-2} + \sqrt{1 - \alpha_t \alpha\_{t-1}} \bar{\boldsymbol{\epsilon}}\_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}\_{t-2} \text{ merges two Gaussians.} \\\
&= \dots \\\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}\\\
q(\mathbf{x}_t \vert \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$

So the sampling of noise and creation of $x_t$ is done in one step only and can be sampled at any timestep.  

## Reverse Diffusion

$q(x\_{t-1} \vert x\_t)$ which denotes the Reverse Process is intractable since statistical estimates of it require computations involving the entire dataset and therefore we need to learn a model $p\_0$ to approximate these conditional probabilities in order to run the reverse diffusion process.

We need to learn a model $p\_0$ to approximate these conditional probabilities

Since $q(x\_{t-1} \vert x\_t)$ will also be Gaussian, for small enough $\beta\_t$, we can choose $p\_0$ to be Gaussian and just parameterize the mean and variance as the **Reverse Diffusion**:

$\quad p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) = \mathcal{N} (\mathbf{x}\_{t-1}; {\mu}\_\theta(\mathbf{x}\_t, t), {\Sigma}\_\theta(\mathbf{x}\_t, t))
$

with
$\mu\_\theta(x\_{t},t)$ as the mean and
$\Sigma\_\theta (x\_{t},t)$ as the variance

conditioned on the noise level $t$ as the to be learned functions of drift and covariance of the Gaussians(by a Neural Net).

As the target image is already defined the problem can be described as a supervised learning problem.


![](/posts/2023_01_11_latent_diffusion_models/images/diffusion-example_pres.png)

*An example of training a diffusion model for modeling a 2D swiss roll data. (Image source: Sohl-Dickstein et al., 2015)*

Hence, our neural network needs to learn/represent the mean and variance. However, the DDPM authors decided to **keep the variance fixed, and let the neural network only learn (represent) the mean $\mu_\theta$ of this conditional probability distribution**.


## Optimization of the Loss Function

To derive an objective function to learn the mean of the backward process, the authors observe that the combination of $q$ and $p\_\theta$ can be seen as a variational auto-encoder (VAE) [(Kingma et al., 2013)](https://arxiv.org/abs/1312.6114).

A Diffusion Model can be trained by finding the reverse Markov transitions that maximize the likelihood of the training data. In practice, training equivalently consists of minimizing the variational upper bound on the negative log likelihood

$- \log p\_\theta(\mathbf{x}\_0)$.

After transformation([Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) for reference), we can write the evidence lower bound (ELBO) as follows:

$$
\begin{aligned}
\- \log p\_\theta(\mathbf{x}_0) 
&\leq - \log p\_\theta(\mathbf{x}_0)+ D\_\text{KL}(q(\mathbf{x}\_{1\:T}\vert\mathbf{x}_0) \vert p\_\theta(\mathbf{x}\_{1\:T}\vert\mathbf{x}_0) )
\end{aligned}
$$


Intuition on the optimization:
For a function $f(x)$, which can't be computed(like e.g. the above negative log-likelihood) and have also a function $g(x)$, which we can compute and fullfills the condition $g(x) <= f(x)$. If we then maximize $g(x)$ we can be certain that $f(x)$ will also increase.


For optimization we use **Kullback-Leibler (KL) Divergences**. 
The KL Divergence is a statistical distance measure of how much one probability distribution $P$ differs from a reference distribution $Q$. 

We are interested in formulating the Loss function in terms of KL divergences because the transition distributions in our Markov chain are Gaussians, and the KL divergence between Gaussians has a closed form. 
For a closer look please look [here](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)


If we rewrite the above Loss function and apply the bayesian rule the upper term can be summarized to a joint probability and will be trainsformed to the Variational Lower Bound:

$$
\begin{aligned}
&= -\log p\_\theta(\mathbf{x}\_0) + \mathbb{E}\_{\mathbf{x}\_{1:T}\sim q(\mathbf{x}\_{1\:T} \vert \mathbf{x}\_0)} \Big[ \log\frac{q(\mathbf{x}\_{1\:T}\vert\mathbf{x}\_0)}{p\_\theta(\mathbf{x}\_{0\:T}) / p\_\theta(\mathbf{x}\_0)} \Big] \\\
&= -\log p\_\theta(\mathbf{x}\_0) + \mathbb{E}\_q \Big[ \log\frac{q(\mathbf{x}\_{1\:T}\vert\mathbf{x}\_0)}{p\_\theta(\mathbf{x}\_{0\:T})} + \log p\_\theta(\mathbf{x}\_0) \Big] \\\
&= \mathbb{E}\_q \Big[ \log \frac{q(\mathbf{x}\_{1\:T}\vert\mathbf{x}\_0)}{p\_\theta(\mathbf{x}\_{0\:T})} \Big] \\\
\text{Let }L\_\text{VLB} 
&= \mathbb{E}\_{q(\mathbf{x}\_{0\:T})} \Big[ \log \frac{q(\mathbf{x}\_{1\:T}\vert\mathbf{x}\_0)}{p\_\theta(\mathbf{x}\_{0\:T})} \Big] \geq - \mathbb{E}\_{q(\mathbf{x}\_0)} \log p\_\theta(\mathbf{x}\_0)
\end{aligned}
$$

Complete calculation can be found [here](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) together with a really nice explanation [here](https://www.youtube.com/watch?v=HoKDTa5jHvg&t=905s)

The objective can be further rewritten to be a combination of several KL-divergence and entropy terms(Detailed process in Appendix B in [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585))

$$
\begin{aligned}
L\_\text{VLB} 
&= \mathbb{E}\_{q(\mathbf{x}\_{0\:T})} \Big[ \log\frac{q(\mathbf{x}\_{1\:T}\vert\mathbf{x}\_0)}{p\_\theta(\mathbf{x}\_{0\:T})} \Big] \\\
&= \dots \\\
&= \mathbb{E}\_q [\underbrace{D\_\text{KL}(q(\mathbf{x}\_T \vert \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_T))}\_{L\_T} + \sum\_{t=2}^T \underbrace{D\_\text{KL}(q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_{t-1} \vert\mathbf{x}\_t))}\_{L\_{t-1}} \underbrace{- \log p\_\theta(\mathbf{x}\_0 \vert \mathbf{x}\_1)}\_{L\_0} ]
\end{aligned}
$$


Reshaped:

$$
\begin{aligned}
L\_\text{VLB} &= L\_T + L\_{T-1} + \dots + L\_0 \\\
\text{where } L\_T &= D\_\text{KL}(q(\mathbf{x}\_T \vert \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_T)) \\\
L\_t &= D\_\text{KL}(q(\mathbf{x}\_t \vert \mathbf{x}\_{t+1}, \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_t \vert\mathbf{x}\_{t+1})) \text{ for }1 \leq t \leq T-1 \\\
L\_0 &= - \log p\_\theta(\mathbf{x}\_0 \vert \mathbf{x}\_1)
\end{aligned}
$$


Every KL term in $L\_\text{VLB}$ except for $L\_0$ compares two Gaussian distributions and therefore they can be computed in closed form. $L\_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $x\_T$ is a Gaussian noise. $L\_t$ formulates the difference between the desired denoising steps and the approximated ones.

It is evident that through the ELBO, maximizing the likelihood boils down to learning the denoising steps $L\_t$.



We would like to train $\boldsymbol{\mu}\_\theta$ to predict $\tilde{\boldsymbol{\mu}}\_t = \frac{1}{\sqrt{\alpha\_t}} \Big( \mathbf{x}\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_t \Big)$.
Because $\mathbf{x}\_t$ is available as input at training time, we can reparameterize the Gaussian noise term instead to make it predict $\boldsymbol{\epsilon}\_t$ from the input  $\mathbf{x}\_t$ at time step $t$:


$$
\begin{aligned}
\boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t) &= \color{red}{\frac{1}{\sqrt{\alpha\_t}} \Big( \mathbf{x}\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t) \Big)} \\\
\text{Thus }\mathbf{x}\_{t-1} &= \mathcal{N}(\mathbf{x}\_{t-1}; \frac{1}{\sqrt{\alpha\_t}} \Big( \mathbf{x}\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t) \Big), \boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t))
\end{aligned}
$$


The loss term $L\_t$ is parameterized to minimize the difference from $\tilde\mu$ :

$$
\begin{aligned}
L\_t 
&= \mathbb{E}\_{\mathbf{x}\_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t) \|^2\_2} \| \color{blue}{\tilde{\boldsymbol{\mu}}\_t(\mathbf{x}\_t, \mathbf{x}\_0)} - \color{green}{\boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t)} \|^2 \Big] \\\
&= \mathbb{E}\_{\mathbf{x}\_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}\_\theta \|^2\_2} \| \color{blue}{\frac{1}{\sqrt{\alpha\_t}} \Big( \mathbf{x}\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha\_t}} \Big( \mathbf{x}\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\boldsymbol{\epsilon}}\_\theta(\mathbf{x}\_t, t) \Big)} \|^2 \Big] \\\
&= \mathbb{E}\_{\mathbf{x}\_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha\_t)^2 }{2 \alpha\_t (1 - \bar{\alpha}\_t) \| \boldsymbol{\Sigma}\_\theta \|^2\_2} \|\boldsymbol{\epsilon}\_t - \boldsymbol{\epsilon}\_\theta(\mathbf{x}\_t, t)\|^2 \Big] \\\
&= \mathbb{E}\_{\mathbf{x}\_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha\_t)^2 }{2 \alpha\_t (1 - \bar{\alpha}\_t) \| \boldsymbol{\Sigma}\_\theta \|^2\_2} \|\boldsymbol{\epsilon}\_t - \boldsymbol{\epsilon}\_\theta(\sqrt{\bar{\alpha}\_t}\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\boldsymbol{\epsilon}\_t, t)\|^2 \Big] 
\end{aligned}
$$


The final objective function $L\_t$ then looks as follows (for a random time step $t$ given $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ) [as shown by Ho et al. (2020)](https://arxiv.org/abs/2006.11239) 

$$ \| \mathbf{\epsilon} - \mathbf{\epsilon}\_\theta(\mathbf{x}\_t, t) \|^2 = \| \mathbf{\epsilon} - \mathbf{\epsilon}\_\theta( \sqrt{\bar{\alpha}\_t} \mathbf{x}\_0 + \sqrt{(1- \bar{\alpha}\_t)  } \mathbf{\epsilon}, t) \|^2.$$

Here, $\mathbf{x}\_0$ is the initial (real, uncorrupted) image, and we see the direct noise level $t$ sample given by the fixed forward process. $\mathbf{\epsilon}$ is the pure noise sampled at time step $t$, and $\mathbf{\epsilon}\_\theta (\mathbf{x}\_t, t)$ is our neural network. The neural network is optimized using a simple mean squared error (MSE) between the true and the predicted Gaussian noise.

Here, $\mathbf{x}\_0$ is the initial (real, uncorrupted) image, and we see the direct noise level $t$ sample given by the fixed forward process. $\mathbf{\epsilon}$ is the pure noise sampled at time step $t$, and $\mathbf{\epsilon}\_\theta (\mathbf{x}\_t, t)$ is our neural network. The neural network is optimized using a simple mean squared error (MSE) between the true and the predicted Gaussian noise.


The training algorithm now looks as follows:

![](/posts/2023_01_11_latent_diffusion_models/images/training_pres.png)



In other words:
* we take a random sample $\mathbf{x}_0$ from the real unknown and possibily complex data distribution $q(\mathbf{x}_0)$
* we sample a noise level $t$ uniformally between $1$ and $T$ (i.e., a random time step)
* we sample some noise from a Gaussian distribution and corrupt the input by this noise at level $t$ (using the nice property defined above)
* the neural network is trained to predict this noise based on the corrupted image $\mathbf{x}_t$ (i.e. noise applied on $\mathbf{x}_0$ based on known schedule $\beta_t$ )

# Neural Nets

The neural network needs to take in a noised image at a particular time step and return the predicted noise. Note that the predicted noise is a tensor that has the same size/resolution as the input image. So technically, the network takes in and outputs tensors of the same shape. What type of neural network can we use for this?

What is typically used here is very similar to that of an Autoencoder, which you may remember from typical "intro to deep learning" tutorials. Autoencoders have a so-called "bottleneck" layer in between the encoder and decoder. The encoder first encodes an image into a smaller hidden representation called the "bottleneck", and the decoder then decodes that hidden representation back into an actual image. This forces the network to only keep the most important information in the bottleneck layer. 


In terms of architecture, the DDPM authors went for a U-Net, introduced by (Ronneberger et al., 2015) (which, at the time, achieved state-of-the-art results for medical image segmentation). This network, like any autoencoder, consists of a bottleneck in the middle that makes sure the network learns only the most important information. Importantly, it introduced residual connections between the encoder and decoder, greatly improving gradient flow (inspired by ResNet in He et al., 2015).

![](/posts/2023_01_11_latent_diffusion_models/images/unet_architecture.jpg)


# The math on Latent Diffusion

It is very slow to generate a sample from DDPM by following the Markov chain of the reverse diffusion process, as 
 can be up to one or a few thousand steps. One data point from Song et al. 2020: “For example, it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on an Nvidia 2080 Ti GPU.”

Latent diffusion model (LDM; Rombach & Blattmann, et al. 2022) runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by first trimming off pixel-level redundancy with autoencoder and then manipulate/generate semantic concepts with diffusion process on learned latent. 

![](/posts/2023_01_11_latent_diffusion_models/images/rombach-latent-space.png)

It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. 

LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by first trimming off pixel-level redundancy with autoencoder and then manipulate/generate semantic concepts with diffusion process on learned latent.

The perceptual compression process relies on an autoencoder model. 

An encoder $\mathcal{E}$ is used to compress the input image $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ to a smaller 2D latent vector $\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}$ , where the downsampling rate $f=H/h=W/w=2^m, m \in \mathbb{N}$. 

Then an decoder $\mathcal{D}$ reconstructs the images from the latent vector, $\tilde{\mathbf{x}} = \mathcal{D}(\mathbf{z})$. 


The paper explored two types of regularization in autoencoder training to avoid arbitrarily high-variance in the latent spaces.
*   KL-reg: A small KL penalty towards a standard normal distribution over the learned latent, similar to [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/).
*   VQ-reg: Uses a vector quantization layer within the decoder, like [VQVAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vq-vae-and-vq-vae-2) but the quantization layer is absorbed by the decoder.


The diffusion and denoising processes happen on the latent vector $\mathbf{z}$. The denoising model is a time-conditioned U-Net, augmented with the cross-attention mechanism to handle flexible conditioning information for image generation (e.g. class labels, semantic maps, blurred variants of an image). 

The design is equivalent to fuse representation of different modality into the model with cross-attention mechanism.

Each type of conditioning information is paired with a domain-specific encoder $\tau_\theta$ to project the conditioning input $y$ to an intermediate representation that can be mapped into cross-attention component, $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$:

$$ 
\begin{aligned} 
&\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\Big) \cdot \mathbf{V} \\\
 &\text{where }\mathbf{Q} = \mathbf{W}^{(i)}_Q \cdot \varphi\_i(\mathbf{z}\_i),\; \mathbf{K} = \mathbf{W}^{(i)}\_K \cdot \tau\_\theta(y),\; \mathbf{V} = \mathbf{W}^{(i)}\_V \cdot \tau\_\theta(y) \\\
  &\text{and } \mathbf{W}^{(i)}\_Q \in \mathbb{R}^{d \times d^i\_\epsilon},\; \mathbf{W}^{(i)}\_K, \mathbf{W}^{(i)}\_V \in \mathbb{R}^{d \times d\_\tau},\; \varphi\_i(\mathbf{z}\_i) \in \mathbb{R}^{N \times d^i\_\epsilon},\; \tau\_\theta(y) \in \mathbb{R}^{M \times d\_\tau} 
\end{aligned}
$$

![](/posts/2023_01_11_latent_diffusion_models/images/rombach-latent-space-comments.jpg)
*Picture from [J. Rafid Siddiqui](https://towardsdatascience.com/what-are-stable-diffusion-models-and-why-are-they-a-step-forward-for-image-generation-aa1182801d46)*


# And what is the result?

![](/posts/2023_01_11_latent_diffusion_models/images/cat_math.jpeg)
*Stable Diffusion "a cat looking at the ocean at sunset"*

![](/posts/2023_01_11_latent_diffusion_models/images/cat_math2.jpeg)
*Stable Diffusion "a cat looking at the ocean at sunset"*


# Ressources

[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

[Illustrations by Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/)

[Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)

[Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752)

[Ho et al. 2020](https://arxiv.org/abs/2006.11239)

[Lilian Weng on Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

[Sergios Karagiannakos on Diffusion Models](https://theaisummer.com/diffusion-models/)

[Hugging Face on Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

[Assembly AI on Diffusion](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)

[J. Rafid Siddiqui on Latent Diffusion](https://towardsdatascience.com/what-are-stable-diffusion-models-and-why-are-they-a-step-forward-for-image-generation-aa1182801d46)

[How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60)

[Stable Diffusion videos from fast.ai](https://www.youtube.com/watch?v=_7rMfsA24Ls&ab_channel=JeremyHoward)
