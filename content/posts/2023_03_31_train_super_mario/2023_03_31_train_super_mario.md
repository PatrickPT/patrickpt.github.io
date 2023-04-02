---
title: "2023_03_31_train_super_mario"
date: 2023-03-31T11:58:22+02:00
draft: False
showToc: true
TocOpen: true
url: /posts/super_mario/
tags: [Fun,Q-Learning,Reinforcement Learning]
---

*WORK IN PROGRESS*

# TLDR;
Learn how to train a Reinforcement Learning Agent to play GameBoy games in a Python written Emulator. With PyBoy, Q-Learning and Super Mario.

# Train your own Agent to play Super Mario

Recently i stumbled upon my old GameBoy an immediately started it and tried to start where i left off 20 years ago.
Unfortunately my hand eye coordination is not what it used to be so i died a few fast deaths.

Thinking about the current possibilities and being lazy i thought about how to write software that could do the job for me and that this would be even more satisfying than playing it myself. I have knonw about the magnitude of recent emulators for all kinds of plattforms and wondered if some of those already contain an API for automating any training.

After some search there it is:
A GameBoy Emulator written in Python which can be fully controlled from a Python Script: [PyBoy](https://github.com/Baekalfen/PyBoy)
(*I can't highlight how big my nerdy enthusiasm for the fact of an emulator in Python is*)

So the place is set up and we have everything we need:
- A GameBoy Emulator which allows us to interact via python
- A Rom of the game you already own (Legally speaking Security Backups are allowed)
- Time and Motivation to start

# The capabilities of PyBoy

## Installation

Assuming that you already have a working Python environment the instalation is super easy.
Just ```pip install pyboy``` and you are ready to go.

## Start

Now if you are just interested in playing a game it is also super easy as you can just start pyboy from your terminal:
``` $ pyboy path/to/your/file.rom ```

PyBoy is loadable as an object in Python. This means, it can be initialized from another script, and be controlled and probed by the scrip:

    from pyboy import PyBoy
    pyboy = PyBoy('ROMs/gamerom.gb')
    while not pyboy.tick():
        pass
    pyboy.stop()

## API

When the emulator is running you can access [PyBoy's API](https://docs.pyboy.dk/index.html).

    from pyboy import WindowEvent

    pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    pyboy.tick() # Process one frame to let the game register the input
    pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)

    pil_image = pyboy.screen_image()
    pil_image.save('screenshot.png')


Now you have everything you need and find all relevant commands in the [PyBoy Documentation](https://docs.pyboy.dk/index.html).


# The Setup for your Reinforcement Learning Algorithm

## Concepts

The setup is following the typical [concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

**Environment** The world that an agent interacts with and learns from.

**Action** $a$ : How the Agent responds to the Environment. The
set of all possible Actions is called *action-space*.

**State** $s$ : The current characteristic of the Environment. The
set of all possible States the Environment can be in is called
*state-space*.

**Reward** $r$ : Reward is the key feedback from Environment to
Agent. It is what drives the Agent to learn and to change its future
action. An aggregation of rewards over multiple time steps is called
**Return**.

**Optimal Action-Value function** $Q^*(s,a)$ : Gives the expected
return if you start in state $s$, take an arbitrary action
$a$, and then for each future time step take the action that
maximizes returns. $Q$ can be said to stand for the “quality” of
the action in a state. We try to approximate this function.

## It's a me Mario

PyBoy is giving an [inspiration](https://github.com/Baekalfen/PyBoy/wiki/Scripts,-AI-and-Bots) on how to set up the Agent for Super Mario.

**Environment** The environment is in this case the world itself: The Floor, Pipes, Blocks the Background and of course the enemies. To capture the evironment means to capture the complete observation space in every single frame.

[! Super Mario Land](images/Super_Mario_Land.webp)

**Action** The actions chosen could be all actions from the actual Window Event: ```List[WindowEvent]``` but for the game it would make absolutely no sense to test the Start Buttone or Select Button. Therefore we focus on LEFT,RIGHT and A.

    baseActions = [WindowEvent.PRESS_ARROW_RIGHT,
                            WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT]

**State** Here it is defined by ```GameState```.

**Reward** The reward is defined by the progress in the game: deaths, time, level and movement.

    clock = current_mario.time_left - prevGameState.time_left
    movement = current_mario.real_x_pos - prevGameState.real_x_pos
    death = -15*(current_mario.lives_left - prevGameState.lives_left)
    levelReward = 15*max((current_mario.world[0] - prevGameState.world[0]), (current_mario.world[1] - prevGameState.world[1])) # +15 if either new level or new world

    reward = clock + death + movement + levelReward

# Let the games begin

I built my own simplified agent which is still work in progress and will be added later. For reference i highly suggest to use the the RL agent built [here](https://github.com/lixado/PyBoy-RL)

# Ressources

[PyBoy Repo](https://github.com/Baekalfen/PyBoy)

[PyBoy Documentation](https://docs.pyboy.dk/index.html)

[Play SNES Super Mario with RL](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

[Example RL for PyBoy](https://github.com/lixado/PyBoy-RL)

