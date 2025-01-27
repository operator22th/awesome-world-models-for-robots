# awesome-world-models-for-robots
## basics
- [World Models](https://www.nvidia.com/en-us/glossary/world-models/)
## benchmark
- arXiv 2024, 03, HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation [Paper](https://arxiv.org/abs/2403.10506) [Website](https://sferrazza.cc/humanoidbench_site/). $15$ whole-body manipulation and $12$ locomotion tasks. This repo contains the code for environments and training.
## dataset
- AgiBot World [Website](https://github.com/OpenDriveLab/AgiBot-World). 1 million+ trajectories from 100 robots.
- LeRobotDataset [Website](https://github.com/huggingface/lerobot). A bunch of models, datasets, and tools for real-world robotics in PyTorch.
- 1xgpt
## models
- Cosmos [Website](https://developer.nvidia.com/cosmos). Autoregressive Video2World/Text2World foundation models.
## toolbox
- Menagerie [Website](https://github.com/google-deepmind/mujoco_menagerie) MuJoCo physics engines. System identification toolbox has not been released.(up to 2025.1)
- MuJoCo Playground [Website](https://playground.mujoco.org/) [Paper](https://playground.mujoco.org/assets/playground_technical_report.pdf) Training environments in mjx. Humanoid Locomotion, Quadruped Locomotion and Manipulation (most robot arms and hand) tasks are included.
## papers
- Daydreamer: World models for physical robot learning
- CoRL 2023 (Oral), Finetuning Offline World Models in the Real World [Website](https://www.yunhaifeng.com/FOWM/) [Paper](https://arxiv.org/abs/2310.16029) Offline pretraining and online finetuning of world models. Robot arm manipulation tasks.
- arxiv 2025, 01, Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics [Paper](https://arxiv.org/abs/2501.10100v1). MBPO sim2real using world models. Quadruped locomotion tasks.
- RSS 2023, Structured World Models from Human Videos. [Paper](https://arxiv.org/abs/2308.10901). Robot arm manipulation tasks.
## related: Robotics & vision-based RL
- CoRL 2022 (Oral), Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion. [Paper](https://arxiv.org/abs/2210.10044).
- CoRL 2022 (Oral), Legged Locomotion in Challenging Terrains using Egocentric Vision. [Paper](https://arxiv.org/pdf/2211.07638).
- CVPR 2023, Affordances from Human Videos as a Versatile Representation for Robotics. [Paper](https://arxiv.org/abs/2304.08488).
- ICML 2023 (Oral), Efficient RL via Disentangled Environment and Agent Representations. [Website](https://sear-rl.github.io/).
- CoRL 2022, VideoDex: Learning Dexterity from Internet Videos. [Website](https://video-dex.github.io/).
- CVPR 2022, Coupling Vision and Proprioception for Navigation of Legged Robots. [Paper](https://arxiv.org/abs/2112.02094).
- CoRL 2024, Continuously Improving Mobile Manipulation with Autonomous Real-World RL. [Paper](https://arxiv.org/abs/2409.20568). Mobile Manipulation.
## related: World Models
- Leo Fan's List. [Website](https://github.com/leofan90/Awesome-World-Models).
- arxiv 2024, 05, Hierarchical World Models as Visual Whole-Body Humanoid Controllers. [Website](https://www.nicklashansen.com/rlpuppeteer/).
- arxiv 2024, 07, Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. [Website](https://boyuan.space/diffusion-forcing/).
## related: Generative models for Decision-Making
- arxiv 2025, 01, Inference-Time Alignment in Diffusion Models with Reward-Guided Generation: Tutorial and Review. [Paper](https://arxiv.org/abs/2501.09685).
- ICML 2024, 3D-VLA: A 3DVision-Language-Action Generative World Model. [Paper](https://arxiv.org/pdf/2403.09631).
