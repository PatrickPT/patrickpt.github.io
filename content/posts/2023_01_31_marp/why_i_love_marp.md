---
title: "How to be more productive when presenting projects"
date: 2023-01-31T09:06:20+01:00
draft: false
math: true
url: /posts/marp/
---

I show a lot of ideas around my code and often i tend to recreate stuff from my documentation. It is often a tedious work, recreating things in Powerpoint. It bores me and somehow it never looks like i want it to look.

Since i started with this website and looked into different ways to represent knowledge to other i started using MARP and ... i love it! It makes me more productive, efficient and faster.

[Hello MARP](https://marp.app)

# What is MARP

Marp (also known as the Markdown Presentation Ecosystem) is a fully open-source tool that let's you transform any Markdown file to a PDF, HTML or PPTX slide deck.

# For whom is MARP the right choice?

It's a rather simple solution capable of basic transformations, figures and tables so it is nothing to create your new super fancy slides for the executive committee. Still it is highly valuable in preparing presentations for your colleagues showing code, math and pictures out of our existing documenation.

It definetly is not for everyone but i like it.

# How can i try it out?

It is super easy. It comes as an extension for [Virtual Studio Code](https://code.visualstudio.com) and can simply be installed and is ready to go from any .md file.

# How can i create slides?
## 1. Create a .md file
Create in any folder an .md file.
## 2. Add frontmatter
You need to add some boilerplate frontmatter.
**frontmatter** is like the configuration of a markdown to add more complex data to the content than just text. It has to be on the top of the file and begins and ends with three dashes ```---```.

It is optional
Each content partial has the following structure ```layout: partial```.
In our case we need to configure the frontmatter to 

```
---
marp: true
---
```
## 2. Create Slides
Slides can be created with the use of dashes.
Each slide ends with three dashes ```---```.
```
---
marp: true
---

# This my first slide

Some content

---

# This is my second slide

Some other content

```

## 2. Export to slide deck

Now you can use the extension and export to a slide of your choosing.

And now you have your first slide deck.

# It is fine but how can i make it more fancy?
You have your first slide deck and now want to further tweak the syle of your presentation?

Configuration is given via comments to each slide.

Overall layout:
```
---
layout: partial
---
```

Layout for the actual slide and all following:
```
<!-- _layout: partial -->
```

## Themes

There are different [Built-in Themes in MARP](https://github.com/marp-team/marp-core/tree/main/themes) which you can use to bring some variety to your slide decks.

They are also fully customizable and you can configure styling for the whole slide deck as well as for only one specific slide.

## Code

It is just as in normal markdown. Code blocks are seperated by ```

```
    ```
    this is my code
    ```
```

## Math

Again like in normal markdown you need to define ```math: true ``` in frontmatter and can put in mathematical formulas with ```$```
```
$x^2$
```
becomes

$x^2$

## Pictures

```
![Caption](link/to/image.jpg)
```

## CSS

Layouts are fully customizable with CSS stylesheets
```
<style>
</style>
```

