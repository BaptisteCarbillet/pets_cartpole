# pets_cartpole


This repo contains an implementation of the algorithm PETS for model based reinforcement learning. The implementation is based on the paper [PETS: Probabilistic Ensembles with Trajectory Sampling](https://arxiv.org/abs/1805.12114) by Chua et al. 

It learns a policy for the CartPole environment using only 26 demonstrations.


### MuJoCo installation
Installation instructions for Mac and Linux can be found [here](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco).



### Conda environment
run the following commands to create the conda environment and activate it
```bash
conda env create --name pets_cartpole --file=environment.yaml
conda activate pets_cartpole
```

### Running the code
To run the code, do the following:
```bash
python -m src.train
```

The resulting policy can be visualized in the notebook 'visualization.ipynb'

