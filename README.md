![image text](materials/image/Fuxi_logo.png)

# **FeverBasketball**

This repository contains an RL environment based on a commercial online basketball game named [Fever Basketball](https://chao.163.com/).
It was created by the [Netease Leihuo Technology](https://leihuo.163.com/) and the [Netease FuXi AI lab](https://fuxi.163.com/) for research purposes.

<div align=center><img src=materials/image/jumpball.png width="25%" height="25%">
<img src=materials/image/pass.png width="25%" height="25%">
<img src=materials/image/dunk.png width="25%" height="25%"></div>

usefull links:
*  Our AAMAS20 Paper (Extended Abstract): [Mastering Basketball with Deep Reinforcement Learning: An Integrated Curriculum Training Approach](https://aamas2020.conference.auckland.ac.nz/extended-abstracts/)
*  Full paper:
*  Blog:
*  Demos: https://www.youtube.com/watch?v=l2SLUydyijs&t=97s

# **Quick Start**

Clone the project to your computer:  
`$ git clone ssh://git@gitlab.leihuo.netease.com:32200/jiahangtian/feverbasketball-opensource.git`  
`$ cd feverbasketball-opensource `  

Run socket server to handle requests of basketball players from game clients:  
`$ python3 ./Manager/run_server.py`  

Start game clinet on computer with windows7/8/10 operation system:  
`1. unzip FeverBasketball.zip`  
`2. cd 'FeverBasketball/selfplay_pack'` or `'/built_in_AI_pack'` to do self-play or play with built-in AI  
`3. double-click 'run_1v1_test.bat'` to start 1v1 game or `'run_3v3_test.bat'` to start 3v3 game (remember to change the ip address to the server's ip first, see [Game setting](materials/doc/settings.md))

# **Train your agents**

Inherit the class of **BaseAgent** (which is a agent with random policy) 
in `./Manager/algorithm/base_agent.py` to define your own agent. 
(Please see the `dqn_agent.py` and `run_server_dqn.py` for example.) 

# **Contents**
*  [Game introduction](#game-introduction)
*  [Game scenarios and rules](#game-scenarios-and-rules)
*  [Game playing](#game-playing)
*  [Game settings](materials/doc/settings.md)
*  [Game observations](materials/doc/observations.md)  
*  [Game actions](materials/doc/actions.md)
*  [Benchmarks](#Benchmarks)

## Game introduction
Fever Basketball simulates a half-court (length=11.4 meters, width=15 meters) basketball match between two teams. 
The game includes the most common basketball aspects, such as jump ball, dribble, three-pointer, dunk, rebound, etc. 
The objective of each team is to score as much as possible to win the match within a limited time.

## Game scenarios and rules
We currently offer two packages (the self-play pack and the built-in AI pack) and three modes (1v1, 2v2 and 3v3) for both game playing and training.
Within each scenario, there are five positions with various attributes and skills (i.e. C, PF, SF, PG, SG) to be chosen from before a match 
(see [Game setting](materials/doc/settings.md)). Taking the 3v3 match for instance, 
one player from each team will do the jump ball first at the beginning of each match. 
The team which gets the ball will be the offense team and the other team will be the defense team. 
The player holding the ball in the offense team is in the *attack* state and the other two players are in the *assist* state. 
Players in the offense team should use offense strategies (such as screen, fast break, jockey for position, etc.) 
and shooting manners (such as jump shot, layup, dunk, etc.) to score. 
For players in the defense team, they should try their best to prevent the offense team from scoring 
by applying defense strategy like one-on-one check, steals, rejections and so on. 
Once the ball is out of the hands of the player who possessing it, all of the players will be in *freeball* state. 
At such a moment, if players of the defense team manage to fetch the ball, 
they have to go through an attack-defense switch process named *ballclear* to regain the possession of the 
ball by dribbling out of the three-point-line. 
Generally speaking, a typical match lasts for three minutes (with an average FPS of 60) except for the overtime. 
We manually extend the length of a match to infinity for the convenience of training purpose.
Besides, players who make the shot clock violation (20 seconds) will be punished by handling the ball possession to the opposite team.

## Game playing
After having started the server by runing `$ python3 ./Manager/run_server.py`, 
you can run the game clinets in different packs to play with the built-in AI or playing with your trained agents.
The key is to set the game clinet args '-p' to number in [0, 1, 2] instead of -1 (see [Game setting](materials/doc/settings.md)), 
which means you can control the first/second/third player in the host team (rendered in color). 

The mapping of the keyboard is as follows. Notice that the keys map to different actions in different scenes.
*  `ARROW UP`: go up
*  `ARROW DOWN`: go down
*  `ARROW LEFT`: go left
*  `ARROW RIGHT`: go right
*  `W`: dribble / defense 
*  `A`: drivein / auto-defense / screen
*  `S`: pass / steal / fetch / request
*  `D`: shoot / block

## Benchmarks
