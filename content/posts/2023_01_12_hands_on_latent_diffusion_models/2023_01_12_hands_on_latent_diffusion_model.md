---
title: "Hands on with Latent Diffusion Models"
date: 2023-01-11T08:28:09Z
draft: true
showToc: true
TocOpen: true
---

**Prerequitises**

To test the models here you need to have an account with HuggingFace [https://huggingface.co/](https://huggingface.co/) - for loading the checkpoint or using the endpoints.
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
The creators of stable-diffusion, an open source latent diffusion model host their model on Huggingface and give access to an endpoint (here called spaces) to test it out:

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

# Stable Diffusion using diffusors

*The following section is based on this [Notebook on Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)*




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

