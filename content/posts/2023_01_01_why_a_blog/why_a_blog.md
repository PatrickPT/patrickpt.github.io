---
title: "Why and how do you create your own AI blog?"
date: 2023-01-06T09:00:00+01:00
draft: False
ShowToc: true
tags: [Medium,Blog,Github,Hugo]
---

Welcome to my blog.

# Why not Medium?

It is 2023 and there are plenty of easy ways to create content about ML, Data Science and AI on the internet. In fact with the accessability of platforms like [Medium](https://medium.com) it is easy.

 *Isn't it actually dumb to create your own blog instead of using these possibilities?*

Maybe, but my purpose is not to attract as many readers as possible but to learn something and make my learnings acceptable for others as well.

*Well, couldn't you have done this also on Medium?*

I did, but as moving further i saw some pitfalls with Medium:

- The writing interface is straightforward and easy to use but it is actually too simple and gives almost no flexibility.
- Medium attract a lot of good writers but also a lot of people just producing content with less quality. If you don't optimize your articles you won't attract any readers. So for my purpose there is no difference in using a provate blog and therefore no advantage in using medium.
- Medium does not support Math Formulas! I could not believe it but there was no decent possibility to include math formulas. a no go in my eyes 


*Ok i get it, Medium is not for you but a blog is a lot of maintenance and effort. Are there any other possibilities.*

- Luckily it is actually quite easy to create your own static site with Github Pages and hugo.
- Also it is a good chance to learn somehting new and make yourself familiar with some web design

# How did you create your blog?

## Which tools are you using for your blog?

As written, i use [Github Pages](https://pages.github.com) and the open-source static site generator [Hugo](https://gohugo.io) which is written in Go. (I chose Hugo without having done a lot of research. There are other generators like e.g. Jekyll but Hugo was open-source, easy to use, blazing fast and free so no need to look further).

## And how did you do actually do it?

1. **Install Hugo**

    
        brew install hugo
    

    If you have a different system than MacOS check the official [installation guide](https://gohugo.io/installation/).

2. **Create a new site in Hugo**
   
   When you decide on a name, think that the name is also the folder and the name of the Repo. To work with Github Pages it needs to have the same name as your git user followed by .github.io

    
        hugo new site <user>.github.io -f yml
    

    I decided to use [PaperMod theme](https://github.com/adityatelange/hugo-PaperMod) and it recommends to use .yml Config instead of .toml - and i am fine with that because i like yaml files.

3. **Install your favorite Hugo Theme**
   
   The project is created but if you try to run it, it will just be an empty page. A style is needed to make it fully functional.

    You could create one from scratch but Hugo has a bunch of themes already prepared and ready to use! You can go to https://themes.gohugo.io/ and choose a theme you like. I like [PaperMod](https://github.com/adityatelange/hugo-PaperMod)

    
        git init
        git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
    

    Adjust your config.yml with **theme: PaperMod**

        baseURL: http://<user>.github.io/
        languageCode: en-us
        title: My New Hugo Site

        theme: PaperMod    

4. **Create Content**
    
    Important is your **content** folder. The folder tree in this folder will reflect your site folder tree.
    Either you create folders and markdown files in those folders manually or you use 

    
        hugo new
    

    e.g.

    
        hugo new /content/posts/first-post/first-post.md
    

    I would recommend to create a folder per post to store your pictures in the same folder as your post they relate to

    
        ---
        heading: "Welcome to my blog"
        subheading: "This is my first-post"
        ---
    

5. **Look at your new site**
   
    With

    
        hugo
    

    you create the html out of your content in your public folder.
    Please keep in mind that for Github Pages you can only choose **docs** as your root directory for your **index.html**.
    Therefore you need to add following config to your **config.yml**

    
        publishDir: docs
    

    With

    
        hugo server -D
    

    you can see the site on localhost with port 1313.
    -D is used to include drafts.

6. **Push to github**
   
   Now you can push your site to github

        git add .
        git commit -m "initial commit"
        git push origin main
   

7. **Configure Github Pages**
   In your repo settings under Pages the root folder needs to be adjusted and your site will hopefully be deployed soon. 

With this short introduction you should be able to set up your own blog really fast and in worst case troubleshoot your way through.
**Enjoy!**

I am happy with my new blog and will further play around with it. If you like my content connect via [LinkedIN](https://www.linkedin.com/in/patrickschnass/).

# References

[1] Github Pages (https://pages.github.com)

[2] Hugo (https://gohugo.io)

[3] PaperMod (https://github.com/adityatelange/hugo-PaperMod)

[4] Markdownguide (https://www.markdownguide.org/basic-syntax/)

[5] Image clickable (https://discourse.gohugo.io/t/how-can-i-make-images-clickable-so-i-can-zoom-them-to-full-screen/34279)

# Further Links

[6] Folder Structure (https://jpdroege.com/blog/hugo-file-organization/)

[7] Markdown with VS Code (https://code.visualstudio.com/docs/languages/markdown)

[8] Trouble with Image paths (https://github.com/adityatelange/hugo-PaperMod/discussions/690)
