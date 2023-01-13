---
title: "Hands on with Latent Diffusion Models"
date: 2023-01-13T09:28:09Z
draft: False
showToc: true
TocOpen: true
---

**Prerequitises**

To test the models here you need to have an account with [HuggingFace](https://huggingface.co/) - for loading the checkpoint or using the endpoints.
Hugging Face is a community and data science platform that provides:
- Tools that enable users to build, train and deploy ML models based on open source (OS) code and technologies.
- A place where a broad community of data scientists, researchers, and ML engineers can come together and share ideas, get support and contribute to open source projects.


# Recap on Latent Diffusion Models

There are mutiple sites and blog posts which explain Latent Diffusion Models including my own [Latent Diffusion Models: What is all the fuzz about?](content/posts/2023_01_11_latent_diffusion_models/2023_01_11_latent_diffusion_models)

To keep it a bit lightweight i can recommend one which explains everything with diagrams(Because i like diagrams for learning).
[Blogpost of Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/)

### I don't understand anything

You don't have any idea what this is all about?

**You can generate beautiful pictures with the help of AI**
All you need to do is create a prompt and enter it into any tool using an algorithm like stable-diffusion which renders your image then.
So
1. Think of a prompt [Examples with prompt search](https://krea.ai//) and
2. Go to [Dall-E](https://openai.com/dall-e-2/)
3. Open an account and try it out.

# NoCode Quickstart

You are not interested in getting your hands dirty?
You don't want to code?
You just want to produce some nice looking images and test your prompt skills?
You are not willing to pay a certain amount to use the capabilities of OpenAI's [Dall-E](https://openai.com/dall-e-2/)?

Then this is for you:

## Prompt Ideas and References

For starters, do you have any idea what you want to create and how to best create your initial prompt?

*Yes* 

Awesome, but as in Google Search: When you try to find the correct search prompt you need to tune the semantics of your thoughts to get what you want:
[How to write stable-diffusion prompts](https://www.howtogeek.com/833169/how-to-write-an-awesome-stable-diffusion-prompt/)

Of course AI can help you with this:
[Prompt Tuning](https://gustavosta-magicprompt-stable-diffusion.hf.space)

*No* 

No worries, you are not the first one to create a prompt and there are already a lot of examples out there:
[Examples with prompt search](https://krea.ai//)
[Atlas on examples with topics](https://atlas.nomic.ai/map/809ef16a-5b2d-4291-b772-a913f4c8ee61/9ed7d171-650b-4526-85bf-3592ee51ea31)

## Use an Endpoint with Stable Diffusion

There are already a few websites giving you access to endpoints for free.
I recommend to use one where you still have access to the codebase of the model and some evaluation.
StabilityAI, the creators of stable-diffusion, an open source latent diffusion model host their model on Huggingface and give access to an endpoint (here called spaces) to test it out:

[Stable Diffusion 2.1 Demo by Stability AI](https://stabilityai-stable-diffusion.hf.space/)

[Model Card of Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2)

## Example

Following prompt:

*"oil painting of a cat sitting on a rainbow"*

becomes after finetuning:

*"oil painting of a cat sitting on a rainbow grass florest, sunset, cliffside ocean scene, diffuse lighting, fantasy, intricate, elegant, highly detailed, lifelike, photorealistic, digital painting, artstation, illustration, concept art, smooth, sharp focus, art by John Collier and Albert Aublet and Krenz Cushart and Artem Demura and Alphonse Mucha"*

and creates this picture with stable-diffusion:
[![cat_rainbow](/posts/2023_01_12_hands_on_latent_diffusion_models/images/cat_rainbow.jpeg)](/posts/2023_01_12_hands_on_latent_diffusion_models/images/cat_rainbow.jpeg)

I like cats!(Like everyone else on the internet i guess)

**Enjoy exploring!**

If you are interested in understanding how to create a Notebook with diffusors please see the following section.

# Stable Diffusion
*...using Hugging Face's `diffusers`*

*The following section focusses on **inference** and is based on [*stable diffusion*](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) and [*diffusers*](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)

*If you want to get a more hands-on guide on **training** diffusion models, please have a look at*
 [*Training with Diffusers*](https://colab.research.google.com/gist/anton-l/f3a8206dae4125b93f05b1f5f703191d/diffusers_training_example.ipynb)

## Summary on diffusers
Stable Diffusion is based on a particular type of diffusion model called **Latent Diffusion**, proposed in [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).

It is created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). It's trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and can run on many consumer GPUs.
See the [model card](https://huggingface.co/CompVis/stable-diffusion) for more information.

However, most of the recent research on diffusion models, e.g. DALL-E 2 and Imagen, is unfortunately not accessible to the broader machine learning community and typically remains behind closed doors.

Here comes Hugging Face's library for diffusion model: [`diffusers`](https://github.com/huggingface/diffusers) with the goals to:

- gather recent diffusion models from independent repositories in a single and long-term maintained project that is built by and for the community,
- reproduce high impact machine learning systems such as DALLE and Imagen in a manner that is accessible for the public, and
create an easy to use API that enables one to train their own models or re-use checkpoints from other repositories for inference.

The core API of `diffusers` is divided into three components:
1. **Pipelines**: high-level classes designed to rapidly generate samples from popular trained diffusion models in a user-friendly fashion.
2. **Models**: popular architectures for training new diffusion models, *e.g.* [UNet](https://arxiv.org/abs/1505.04597).
3. **Schedulers**: various techniques for generating images from noise during *inference* as well as to generate noisy images for *training*.

## Create your own

### Install diffusers

```
!pip install diffusers==0.11.0
!pip install transformers scipy ftfy accelerate
!pip install "ipywidgets>=7,<8"
!pip install safetensors
```

### Input your Hugging Face Token

As mentioned earlier you need a token with huggingface to import the pretrained snapshots

```
from huggingface_hub import notebook_login
notebook_login()
```

### Create Pipeline

`StableDiffusionPipeline` is an end-to-end inference pipeline that you can use to generate images from text with just a few lines of code.

First, we load the pre-trained weights of all components of the model. Here we use Stable Diffusion version 2.1 ([stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)), but there are other variants that you may want to try:
* [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
* [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
* [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1). This version can produce images with a resolution of 768x768, while the others work at 512x512.

This stable-diffusion-2-1 model is fine-tuned from stable-diffusion-2 (768-v-ema.ckpt) with an additional 55k steps on the same dataset (with punsafe=0.1), and then fine-tuned for another 155k extra steps with punsafe=0.98.

In addition to the model id [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1), we're also passing a specific `torch_dtype` to the `from_pretrained` method.

The weights are loaded from the half-precision branch [`fp16`](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/fp16) and we need to tell `diffusers` to expect the weights in float16 precision by passing `torch_dtype=torch.float16`.

We can import the `DDPMPipeline`, which will allow you to do inference with a couple of lines of code.
The `from_pretrained()` method allows downloading the model and its configuration from [the Hugging Face Hub](https://huggingface.co/stabilityai/stable-diffusion-2-1), a repository of over 60,000 models shared by the community.

```
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

```

### Generate Image

To generate an image, we simply run the pipeline and don't even need to give it any input, it will generate a random initial noise sample and then iterate the diffusion process. Here we use the inital prompt from above

The pipeline returns as output a dictionary with a generated `sample` of interest.

```
prompt = "oil painting of a cat sitting on a rainbow grass florest, sunset, cliffside ocean scene, diffuse lighting, fantasy, intricate, elegant, highly detailed, lifelike, photorealistic, digital painting, artstation, illustration, concept art, smooth, sharp focus, art by John Collier and Albert Aublet and Krenz Cushart and Artem Demura and Alphonse Mucha"
image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

image.save(f"astronaut_rides_horse.png")
```
Et voila

[![cat_rainbow_stable_diffusion](/posts/2023_01_12_hands_on_latent_diffusion_models/images/cat_sitting_rainbow.png)](/posts/2023_01_12_hands_on_latent_diffusion_models/images/cat_sitting_rainbow.png)



# References
[Latent Diffusion Models: What is all the fuzz about?](content/posts/2023_01_11_latent_diffusion_models/2023_01_11_latent_diffusion_models)

[Hugging Face](https://huggingface.co/)

[Blogpost of Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/)

[Dall-E](https://openai.com/dall-e-2/)

[Examples with prompt search](https://krea.ai//)

[Atlas on examples with topics](https://atlas.nomic.ai/map/809ef16a-5b2d-4291-b772-a913f4c8ee61/9ed7d171-650b-4526-85bf-3592ee51ea31)

[How to write stable-diffusion prompts](https://www.howtogeek.com/833169/how-to-write-an-awesome-stable-diffusion-prompt/)

[Prompt Tuning](https://gustavosta-magicprompt-stable-diffusion.hf.space)

# Further Links
[What's HuggingFace on Medium](https://towardsdatascience.com/whats-hugging-face-122f4e7eb11a)

