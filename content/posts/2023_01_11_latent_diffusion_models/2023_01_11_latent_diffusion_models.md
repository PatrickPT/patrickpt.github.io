---
title: "Latent Diffusion Models: What is all the fuzz about?"
date: 2023-01-10T14:50:18+01:00
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
Let me briefly introduce you to Latent Diffusion Models and explain the math.

If you are interested in a Hands-On you can find that in my other post:

[Hands on Latent Diffusion Models](content/posts/2023_01_12_hands_on_latent_diffusion_models/2023_01_12_hands_on_latent_diffusion_model)

# What is this post about:

A brand-new category of cutting-edge generative models called diffusion models generate a variety of high-resolution images.
They have already received a great deal of attention as a result of OpenAI, Nvidia, and Google's success in training massive models.
We'll take a deeper look into **Denoising Diffusion Probabilistic Models** (also known as DDPMs, diffusion models, score-based generative models or simply [autoencoders](https://benanne.github.io/2022/01/31/diffusion.html)) as researchers have been able to achieve remarkable results with them for (un)conditional image/audio/video generation. Popular examples (at the time of writing) include [GLIDE](https://arxiv.org/abs/2112.10741) and [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI, [Latent Diffusion](https://github.com/CompVis/latent-diffusion) by the University of Heidelberg and [ImageGen](https://imagen.research.google/) by Google Brain.

The idea of diffusion for generative modeling was introduced in ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)). However, it took until ([Song et al., 2019](https://arxiv.org/abs/1907.05600)) (at Stanford University), and then ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) (at Google Brain) who independently improved the approach.
DDPM originally was introduced in a paper by ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)), and further anaylzed in Phil Wang's [implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) - which itself is based on the [original TensorFlow implementation](https://github.com/hojonathanho/diffusion). 

There are [several perspectives](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw) on diffusion models. Here, the discrete-time (latent variable model) perspective is analyzed.

# Intuition on Latent Diffusion Models

You may ask "What is the intuition behind latent diffusion models?"
Let's break it down with a short example to make it clear:
You are a painter hired by Vatican with the task to repaint the fresco at the ceiling of [sixtinian chapel](https://de.wikipedia.org/wiki/Sixtinische_Kapelle#/media/Datei:CAPPELLA_SISTINA_Ceiling.jpg).

The requirement is to recreate the fresco with pictures of cats.
Most likely you will be overwhelmed with that task and immediately think of structuring your work into smaller chunks. You want to start with one dedicated part of the whole fresco, for example the creation of Adam. You will put some effort in it, first paint the background of that chunk of the painting. Then you may outline the structures you want to paint, then you start with one object like the arm of Adam,  and then focus on the hand. Finally you are happy with the scene, decide to finish it and tackle the next part. Still later you may change things after you decided that it fits better with the overall fresco.

(#TODO Picture creation of Adam with Cats)

Finally you are building a masterpiece out of masterpieces lasting centuries until someone thinks dogs are nicer than cats(so never).
The same idea comes with Diffusion: Decompose the image to smaller chunks and create gradually the best sample possible in many small steps. Breaking up the image sampling allows the models to correct itself over those small steps iteratively and produce a good sample.

Unfortunately nothing is cost-neutral. Like it will cost a painter like Michealangelo almost 4 years to finish the fresco in Sixtinian chapel, the iterative process makes the Models slow at sampling(At least compared to GANs).

# What are Latent Diffusion Models?

Focussing on the process of Denoising diffusion probabilistic models and to wrap it up: Latent Diffusion Models are a type of generative model that can be used to reconstruct an input signal from a noisy version of that signal. These models are based on the concept of diffusion, which refers to the way in which a signal spreads out and becomes more diffuse over time. 

If you compare it to other generative models such as Normalizing Flows, GANs or VAEs: They all convert noise from some simple distribution to a data sample. This is also the case with Latent Diffusion Models where **a neural network learns to gradually denoise data** starting from pure noise.

[![types_gans](/posts/2023_01_11_latent_diffusion_models/images/types_gans.png)](/posts/2023_01_11_latent_diffusion_models/images/types_gans.png)
*Overview of the different types of generative [models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/HowdoestheDiffusionprocesswork?)*

At a high level, they work by first representing the input signal as a set of latent variables, which are then transformed through a series of probabilistic transformations to produce the output signal. The transformation process is designed to smooth out the noise in the input signal and reconstruct a cleaner version of the signal.

Transformation consist of 2 processes. 

In a bit more detail for images, the set-up consists of 2 processes:
* a fixed (or predefined) forward diffusion process \\(q\\) of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise
* a learned reverse denoising diffusion process \\(p_\theta\\), where a neural network is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.


# I want to see the math

Both the forward and reverse process indexed by \\(t\\) happen for some number of finite time steps \\(T\\) . You start with \\(t=0\\) where you sample a real image \\(\mathbf{x}_0\\) from your data distribution (let's say an image of a cat from ImageNet), and the forward process samples some noise from a Gaussian distribution at each time step \\(t\\), which is added to the image of the previous time step. Given a sufficiently large \\(T\\) and a well behaved schedule for adding noise at each time step, you end up with what is called an [isotropic Gaussian distribution](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) at \\(t=T\\) via a gradual process.  Isotropic means the probability density is equal (iso) in every direction (tropic). In gaussians this can be achieved with a \\(\sigma^2 I\\) covariance matrix

## In more mathematical form

A tractable loss function top optimize the neural net is needed.

Let \\(q(\mathbf{x}_0)\\) be the real data distribution, say of "real images". We can sample from this distribution to get an image, \\(\mathbf{x}_0 \sim q(\mathbf{x}_0)\\). We define the forward diffusion process \\(q(\mathbf{x}_t | \mathbf{x}_{t-1})\\) which adds Gaussian noise at each time step \\(t\\), according to a known variance schedule \\(0 < \beta_1 < \beta_2 < ... < \beta_T < 1\\) as
$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}). 
$$

Recall that a normal distribution (also called Gaussian distribution) is defined by 2 parameters: a mean \\(\mu\\) and a variance \\(\sigma^2 \geq 0\\). Basically, each new (slightly noisier) image at time step \\(t\\) is drawn from a **conditional Gaussian distribution** with \\(\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}\\) and \\(\sigma^2_t = \beta_t\\), which we can do by sampling \\(\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\\) and then setting \\(\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\epsilon}\\). 

Note that the \\(\beta_t\\) aren't constant at each time step \\(t\\) (hence the subscript) --- in fact one defines a so-called **"variance schedule"**, which can be linear, quadratic, cosine, etc. as we will see further (a bit like a learning rate schedule). 

So starting from \\(\mathbf{x}_0\\), we end up with \\(\mathbf{x}_1,  ..., \mathbf{x}_t, ..., \mathbf{x}_T\\), where \\(\mathbf{x}_T\\) is pure Gaussian noise if we set the schedule appropriately.

Now, if we knew the conditional distribution \\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\), then we could run the process in reverse: by sampling some random Gaussian noise \\(\mathbf{x}_T\\), and then gradually "denoise" it so that we end up with a sample from the real distribution \\(\mathbf{x}_0\\).

However, we don't know \\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\). It's intractable since it requires knowing the distribution of all possible images in order to calculate this conditional probability. Hence, we're going to leverage a neural network to **approximate (learn) this conditional probability distribution**, let's call it \\(p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)\\), with \\(\theta\\) being the parameters of the neural network, updated by gradient descent. 

Ok, so we need a neural network to represent a (conditional) probability distribution of the backward process. If we assume this reverse process is Gaussian as well, then recall that any Gaussian distribution is defined by 2 parameters:
* a mean parametrized by \\(\mu_\theta\\);
* a variance parametrized by \\(\Sigma_\theta\\);

so we can parametrize the process as 
$$ p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))$$
where the mean and variance are also conditioned on the noise level \\(t\\).

Hence, our neural network needs to learn/represent the mean and variance. However, the DDPM authors decided to **keep the variance fixed, and let the neural network only learn (represent) the mean \\(\mu_\theta\\) of this conditional probability distribution**. From the paper:

> First, we set \\(\Sigma_\theta ( \mathbf{x}_t, t) = \sigma^2_t \mathbf{I}\\) to untrained time dependent constants. Experimentally, both \\(\sigma^2_t = \beta_t\\) and \\(\sigma^2_t  = \tilde{\beta}_t\\) (see paper) had similar results. 
This was then later improved in the [Improved diffusion models](https://openreview.net/pdf?id=-NEXDKk8gZ) paper, where a neural network also learns the variance of this backwards process, besides the mean.

So we continue, assuming that our neural network only needs to learn/represent the mean of this conditional probability distribution.

## Defining an objective function (by reparametrizing the mean)

To derive an objective function to learn the mean of the backward process, the authors observe that the combination of \\(q\\) and \\(p_\theta\\) can be seen as a variational auto-encoder (VAE) [(Kingma et al., 2013)](https://arxiv.org/abs/1312.6114). Hence, the **variational lower bound** (also called ELBO) can be used to minimize the negative log-likelihood with respect to ground truth data sample \\(\mathbf{x}_0\\) (we refer to the VAE paper for details regarding ELBO). It turns out that the ELBO for this process is a sum of losses at each time step \\(t\\), \\(L = L_0 + L_1 + ... + L_T\\). By construction of the forward \\(q\\) process and backward process, each term (except for \\(L_0\\)) of the loss is actually the **KL divergence between 2 Gaussian distributions** which can be written explicitly as an L2-loss with respect to the means!

A direct consequence of the constructed forward process \\(q\\), as shown by Sohl-Dickstein et al., is that we can sample \\(\mathbf{x}_t\\) at any arbitrary noise level conditioned on \\(\mathbf{x}_0\\) (since sums of Gaussians is also Gaussian). This is very convenient:  we don't need to apply \\(q\\) repeatedly in order to sample \\(\mathbf{x}_t\\). 
We have that 
$$q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})$$

with \\(\alpha_t := 1 - \beta_t\\) and \\(\bar{\alpha}_t := \Pi_{s=1}^{t} \alpha_s\\). Let's refer to this equation as the "nice property". This means we can sample Gaussian noise and scale it appropriatly and add it to \\(\mathbf{x}_0\\) to get \\(\mathbf{x}_t\\) directly. Note that the \\(\bar{\alpha}_t\\) are functions of the known \\(\beta_t\\) variance schedule and thus are also known and can be precomputed. This then allows us, during training, to **optimize random terms of the loss function \\(L\\)** (or in other words, to randomly sample \\(t\\) during training and optimize \\(L_t\\)).

Another beauty of this property, as shown in Ho et al. is that one can (after some math, for which we refer the reader to [this excellent blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)) instead **reparametrize the mean to make the neural network learn (predict) the added noise (via a network \\(\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)\\)) for noise level \\(t\\)** in the KL terms which constitute the losses. This means that our neural network becomes a noise predictor, rather than a (direct) mean predictor. The mean can be computed as follows:

$$ \mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(  \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$

The final objective function \\(L_t\\) then looks as follows (for a random time step \\(t\\) given \\(\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\\) ): 

$$ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 = \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\epsilon}, t) \|^2.$$

Here, \\(\mathbf{x}_0\\) is the initial (real, uncorrupted) image, and we see the direct noise level \\(t\\) sample given by the fixed forward process. \\(\mathbf{\epsilon}\\) is the pure noise sampled at time step \\(t\\), and \\(\mathbf{\epsilon}_\theta (\mathbf{x}_t, t)\\) is our neural network. The neural network is optimized using a simple mean squared error (MSE) between the true and the predicted Gaussian noise.

In other words:
* we take a random sample \\(\mathbf{x}_0\\) from the real unknown and possibily complex data distribution \\(q(\mathbf{x}_0)\\)
* we sample a noise level \\(t\\) uniformally between \\(1\\) and \\(T\\) (i.e., a random time step)
* we sample some noise from a Gaussian distribution and corrupt the input by this noise at level \\(t\\) (using the nice property defined above)
* the neural network is trained to predict this noise based on the corrupted image \\(\mathbf{x}_t\\) (i.e. noise applied on \\(\mathbf{x}_0\\) based on known schedule \\(\beta_t\\))

In reality, all of this is done on batches of data, as one uses stochastic gradient descent to optimize neural networks.

## The neural network

The neural network needs to take in a noised image at a particular time step and return the predicted noise. Note that the predicted noise is a tensor that has the same size/resolution as the input image. So technically, the network takes in and outputs tensors of the same shape. What type of neural network can we use for this? 

What is typically used here is very similar to that of an [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder), which you may remember from typical "intro to deep learning" tutorials. Autoencoders have a so-called "bottleneck" layer in between the encoder and decoder. The encoder first encodes an image into a smaller hidden representation called the "bottleneck", and the decoder then decodes that hidden representation back into an actual image. This forces the network to only keep the most important information in the bottleneck layer.

In terms of architecture, the DDPM authors went for a **U-Net**, introduced by ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) (which, at the time, achieved state-of-the-art results for medical image segmentation). This network, like any autoencoder, consists of a bottleneck in the middle that makes sure the network learns only the most important information. Importantly, it introduced residual connections between the encoder and decoder, greatly improving gradient flow (inspired by ResNet in [He et al., 2015](https://arxiv.org/abs/1512.03385)).

[![unet](/posts/2023_01_12_hands_on_latent_diffusion_models/images/unet-model.png)](/posts/2023_01_12_hands_on_latent_diffusion_models/images/unet-model.png)


As can be seen, a U-Net model first downsamples the input (i.e. makes the input smaller in terms of spatial resolution), after which upsampling is performed.

# And now?

[![unet](/posts/2023_01_12_hands_on_latent_diffusion_models/images/stable_diffusion.png)](/posts/2023_01_12_hands_on_latent_diffusion_models/images/stable_diffusion.png)