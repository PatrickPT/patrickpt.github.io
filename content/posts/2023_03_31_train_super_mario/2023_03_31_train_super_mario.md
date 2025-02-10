---
title: "Play Super Mario with RL"
date: 2023-03-31T11:58:22+02:00
draft: false
summary: Learn how to train a Reinforcement Learning Agent to play GameBoy games in a Python written Emulator. With PyBoy, Q-Learning and Super Mario.
showToc: true
TocOpen: true
math: true
url: /posts/super_mario/
tags: [Fun,Q-Learning,Reinforcement Learning]
---

# TL;DR
Learn how to train a Reinforcement Learning Agent to play GameBoy games in a Python written Emulator. With PyBoy, Q-Learning and Super Mario.

# Train your own RL Agent to play Super Mario

![Super Mario Land](/posts/2023_03_31_train_super_mario/images/Super_Mario_Land.jpeg)

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

## Concepts of RL

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The goal of RL is to find an optimal policy that maximizes a long-term reward signal. The agent takes actions in the environment and receives feedback in the form of rewards or penalties. The agent then updates its policy based on the feedback received, in order to maximize the total reward it receives over time.

Our setup is following the typical [concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

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

## Specification of Q Learning

Q-Learning is a popular reinforcement learning algorithm that learns the optimal action-value function, also known as the Q-function, for a given environment. The Q-function represents the expected long-term reward for taking a particular action in a given state. The Q-learning algorithm updates the Q-function based on the rewards received and the estimated Q-values for the next state. The agent then uses the updated Q-function to select the next action.

The Q-learning algorithm is based on the Bellman equation, which is a recursive equation that expresses the optimal Q-value in terms of the expected reward for the current action and the expected Q-value for the next state. The Q-learning algorithm uses a greedy approach to select actions, meaning that it always chooses the action with the highest estimated Q-value.

Q-learning stores the results between iterations in a Q-table, which is essentially a lookup table that contains the expected reward values for every state-action pair in the environment. The Q-table is initialized with arbitrary values and is updated over time as the agent interacts with the environment.

At each iteration, the agent observes the current state of the environment, selects an action based on the Q-values in the Q-table, performs the action, and observes the reward and the next state. The agent then updates the Q-value for the state-action pair based on the Bellman equation, which expresses the optimal Q-value for a state-action pair as the sum of the immediate reward and the discounted expected future reward.

The Q-value update equation is as follows:

Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))

where:

Q(s,a) is the current Q-value for the state-action pair (s,a)
alpha is the learning rate, which determines how much the Q-value is updated based on the new information
r is the immediate reward received for taking the action a in state s
gamma is the discount factor, which determines how much weight is given to future rewards
max(Q(s',a')) is the maximum Q-value for the next state s' and all possible actions a' that can be taken from s'
After the Q-value update, the agent moves to the next state and repeats the process until it reaches the terminal state.

As the agent continues to interact with the environment, the Q-table is gradually updated with more accurate estimates of the optimal Q-values. The agent uses the Q-table to select the optimal action in each state, based on the highest Q-value in the table for that state. By learning from experience and updating the Q-values over time, Q-learning allows the agent to make better decisions and maximize the cumulative reward it receives from the environment.

## It's a me Mario

PyBoy is giving an [inspiration](https://github.com/Baekalfen/PyBoy/wiki/Scripts,-AI-and-Bots) on how to set up the Agent for Super Mario.

**Environment** The environment is in this case the world itself: The Floor, Pipes, Blocks the Background and of course the enemies. To capture the evironment means to capture the complete observation space in every single frame.

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

I built my own simplified agent on following example and had a ton of fun [here](https://github.com/lixado/PyBoy-RL)

# Ressources

[PyBoy Repo](https://github.com/Baekalfen/PyBoy)

[PyBoy Documentation](https://docs.pyboy.dk/index.html)

[Play SNES Super Mario with RL](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

[Example RL for PyBoy](https://github.com/lixado/PyBoy-RL)

