# Multi Agent Allocation with Generative Network

## Recommended system
Recommended system (tested):
- Windows/Ubuntu
- Python 3.8.13

Python packages used by the example provided and their recommended version:
- pytorch==1.7.1
- gym==0.21.0
- numpy==1.22.3
- stable-baselines3==1.5.1
- tensorboard==2.8.0
- scipy==1.8.0
- pandas==1.3.0
- pyglet==1.5.23

## Training
You can configure some settings in `train.py` and `curriculum_approaches.py` (number of episodes, evaluation interval) and the other settings (reward structure, maximum allowable prizes, collision dynamics etc.) in `point_mass_formation.py`

## Test
You can configure some settings in `test.py`
![](https://github.com/AkgunOnur/Multi-Agent-Allocation-with-Generative-Network/blob/curriculum/gym_anim.gif)
