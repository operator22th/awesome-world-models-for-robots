# awesome-world-models-for-robots
## overview
- [World Models](https://www.nvidia.com/en-us/glossary/world-models/)
- arxiv 2024, 11, Understanding World or Predicting Future? A Comprehensive Survey of World Models [Paper](https://arxiv.org/pdf/2411.14499).
## benchmark
- arXiv 2024, 03, HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation [Paper](https://arxiv.org/abs/2403.10506) [Website](https://sferrazza.cc/humanoidbench_site/). $15$ whole-body manipulation and $12$ locomotion tasks. This repo contains the code for environments and training.
## dataset
- Pysical AI [Website](https://huggingface.co/collections/nvidia/physical-ai-67c643edbb024053dcbcd6d8).
- AgiBot World [Website](https://github.com/OpenDriveLab/AgiBot-World). 1 million+ trajectories from 100 robots.
- LeRobotDataset [Website](https://github.com/huggingface/lerobot). A bunch of models, datasets, and tools for real-world robotics in PyTorch.
- 1xgpt [Website](https://github.com/1x-technologies/1xgpt).
- OXE [Paper](https://arxiv.org/abs/2310.08864).
## models
- V-JEPA [Website](https://ai.meta.com/vjepa/). [Paper](https://arxiv.org/abs/2506.09985) [Code](https://github.com/facebookresearch/vjepa2).
- Cosmos [Website](https://developer.nvidia.com/cosmos). [Paper](https://arxiv.org/abs/2501.03575). Autoregressive Video2World/Text2World foundation models.
## toolbox
- Menagerie [Website](https://github.com/google-deepmind/mujoco_menagerie) MuJoCo physics engines. System identification toolbox has not been released.(up to 2025.1)
- MuJoCo Playground [Website](https://playground.mujoco.org/) [Paper](https://playground.mujoco.org/assets/playground_technical_report.pdf) Training environments in mjx. Humanoid Locomotion, Quadruped Locomotion and Manipulation (most robot arms and hand) tasks are included.
## papers
- arxiv 2025, 05, Evaluating Robot Policies in a World Model. [Paper](https://arxiv.org/abs/2506.00613).
- arxiv 2025, 05, RLVR-World: Training World Models with Reinforcement Learning. [Paper](https://arxiv.org/abs/2505.13934).
- ICML 2025, Trajectory World Models for Heterogeneous Environments. [Paper](https://arxiv.org/abs/2502.01366).
- Neulps 2024, iVideoGPT: Interactive VideoGPTs are Scalable World Models. [Website](https://thuml.github.io/iVideoGPT/). [Code](https://github.com/thuml/iVideoGPT).
- RSS 2025, Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets. [Website](https://weirdlabuw.github.io/uwm/).
- RSS 2025, Unified Video Action Model. [Website](https://unified-video-action-model.github.io/).
- ICRA 2025, World Model-based Perception for Visual Legged Locomotion. [Website](https://wmp-loco.github.io/). [Code](https://github.com/bytedance/WMP).
- arxiv 2025, 03, Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning. [Paper](https://arxiv.org/abs/2503.01837). [Website](https://adrialopezescoriza.github.io/demo3/).
- ICRA 2024, MoDem-V2: Visuo-Motor World Models for Real-World Robot Manipulation. [Paper](https://arxiv.org/pdf/2309.14236).
- CoRL 2024, Multi-Task Interactive Robot Fleet Learning with Visual World Models. [Paper](https://arxiv.org/pdf/2410.22689). Visual world model for anomaly detection.
- CVPR 2023, Affordances from Human Videos as a Versatile Representation for Robotics. [Paper](https://arxiv.org/abs/2304.08488). Prediction contact points and trajectory waypoints, then use it for downstream tasks (suitable for different learning paradigms).
- RSS 2023, Structured World Models from Human Videos. [Paper](https://arxiv.org/abs/2308.10901). Robot arm manipulation tasks. World Models with structured action space design.
- RSS 2024, HRP: Human Affordances for Robotic Pre-Training. [Paper](https://arxiv.org/abs/2407.18911).
- arxiv 2025, 02, Strengthening Generative Robot Policies through Predictive World Modeling. [Paper](https://arxiv.org/pdf/2502.00622). Stengthen imitation learning with world model.
- arxiv 2024, 11, DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. [Website](https://dino-wm.github.io/). World model for MPC. DINOv2 for representation.
- CoRL 2022, Daydreamer: World models for physical robot learning. [Paper](https://arxiv.org/abs/2206.14176).
- CoRL 2023 (Oral), Finetuning Offline World Models in the Real World [Website](https://www.yunhaifeng.com/FOWM/) [Paper](https://arxiv.org/abs/2310.16029) Offline pretraining and online finetuning of world models. Robot arm manipulation tasks.
- arxiv 2025, 01, Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics [Paper](https://arxiv.org/abs/2501.10100v1). MBPO sim2real using world models. Quadruped locomotion tasks.
- ICLR 2024 (Outstanding Paper), UniSim: Learning Interactive Real-World Simulators [Website](https://universal-simulator.github.io/unisim/).
- ICLR 2024, Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation [Website](https://gr1-manipulation.github.io/).
## workshop
- ICML 2024, Multi-modal Foundation Model meets Embodied AI [Website](https://icml-mfm-eai.github.io/).
- ICLR 2025, Generative Models for Robot Learning. [Website](https://genbot-workshop.github.io/).
- ICLR 2025, World Models. [Website](https://sites.google.com/view/worldmodel-iclr2025/).
- ICML 2025, Building Physically Plausible World Models. [Website](https://physical-world-modeling.github.io/).
## related: World Models
- Leo Fan's List. [Website](https://github.com/leofan90/Awesome-World-Models).
- arxiv 2025, 03, Denoising Hamiltonian Network for Physical Reasoning. [Paper](https://arxiv.org/abs/2503.07596).
- arxiv 2024, 05, Hierarchical World Models as Visual Whole-Body Humanoid Controllers. [Website](https://www.nicklashansen.com/rlpuppeteer/).
- ICML 2024, Offline Transition Modeling via Contrastive Energy Learning. [Code](https://github.com/Ruifeng-Chen/Energy-Transition-Models).
- ICML 2024, 3D-VLA: A 3DVision-Language-Action Generative World Model. [Paper](https://arxiv.org/pdf/2403.09631).
- ICML 2024 (Oral), Genie: Generative Interactive Environments. [Paper](https://arxiv.org/abs/2402.15391).
- ICML 2024 (Oral), Learning to Model the World with Language. [Paper](https://arxiv.org/abs/2308.01399). [Website](https://dynalang.github.io/).
- 2024, 12, Genie2 [Blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/).
- arxiv 2025, 03, PISA Experiments: Exploring Physics Post-Training for Video Diffusion Models by Watching Stuff Drop. [Webstite](https://vision-x-nyu.github.io/pisa-experiments.github.io/).
## related: LLM as WM
- ICLR 2025, Monte Carlo Planning with Large Language Model for Text-Based Games. [Paper](https://openreview.net/forum?id=r1KcapkzCt).
- arxiv 2024, AgentGym: Evolving Large Language Model-based Agents across Diverse Environments. [Paper](https://arxiv.org/abs/2406.04151). [Code](https://github.com/WooooDyy/AgentGym).
- NIPS 2023, Language Models Meet World Models: Embodied Experiences Enhance Language Models. [Paper](https://arxiv.org/abs/2305.10626). [Openreview](https://openreview.net/forum?id=SVBR6xBaMl).
- NIPS 2023, Large Language Models as Commonsense Knowledge for Large-Scale Task Planning. [Website](https://llm-mcts.github.io/). [Paper](https://arxiv.org/abs/2305.14078).
- NIPS 2023, ChessGPT: Bridging Policy Learning and Language Modeling. [Paper](https://arxiv.org/abs/2306.09200). [Code](https://github.com/waterhorse1/ChessGPT).
## related: Transfer Learning
- arxiv 2022, 01, Transferability in Deep Learning: A Survey. [Paper](https://arxiv.org/abs/2201.05867).
## related: Robotics & Foundation models
- 2025, 03,  GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. [Code](https://github.com/NVIDIA/Isaac-GR00T).
- RSS 2024, OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics. [Paper](https://arxiv.org/pdf/2401.12202).
- CoRL 2023 (Oral), VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models. [Website](https://voxposer.github.io/).
- ICLR 2024, Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models. [Paper](https://arxiv.org/abs/2310.10639).
- ICRA 2025, WildLMA: Long Horizon Loco-MAnipulation in the Wild. [Website](https://wildlma.github.io/).
- arxiv, 2024, 12, NaVILA: Legged Robot Vision-Language-Action Model for Navigation. [Website](https://navila-bot.github.io/).
- arxiv, 2024, 10, GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs. [Paper](https://arxiv.org/pdf/2410.03645).
## related: Robotics & Vision-based RL
- CoRL 2022 (Oral), Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion. [Paper](https://arxiv.org/abs/2210.10044).
- CoRL 2022 (Oral), Legged Locomotion in Challenging Terrains using Egocentric Vision. [Paper](https://arxiv.org/pdf/2211.07638).
- ICML 2023 (Oral), Efficient RL via Disentangled Environment and Agent Representations. [Website](https://sear-rl.github.io/).
- CoRL 2022, VideoDex: Learning Dexterity from Internet Videos. [Website](https://video-dex.github.io/).
- CVPR 2022, Coupling Vision and Proprioception for Navigation of Legged Robots. [Paper](https://arxiv.org/abs/2112.02094).
- CoRL 2024, Continuously Improving Mobile Manipulation with Autonomous Real-World RL. [Paper](https://arxiv.org/abs/2409.20568). Mobile Manipulation.
- RSS 2023, Pre-Training for Robots: Offline RL Enables Learning New Tasks from a Handful of Trials. [Paper](https://arxiv.org/abs/2210.05178).
- CoRL 2024, Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance. [Website](https://nakamotoo.github.io/V-GPS/).
## related: Robotics & Visual representations
- NeurIPS 2024, DynaMo: In-Domain Dynamics Pretraining for Visuo-Motor Control. [Paper](https://arxiv.org/pdf/2409.12192).
- RSS 2024, HRP: Human Affordances for Robotic Pre-Training. [Paper](https://arxiv.org/abs/2407.18911).
- ICML 2023 (Oral), Efficient RL via Disentangled Environment and Agent Representations. [Website](https://sear-rl.github.io/).
- CVPR 2023, Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. [Paper](https://arxiv.org/abs/2301.08243).
- ICML 2022, On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline. [Paper](https://arxiv.org/abs/2212.05749).
- IROS 2023, Visual Reinforcement Learning with Self-Supervised 3D Representations. [Paper](https://arxiv.org/abs/2210.07241).
## related: Generative models for Decision-Making
- arxiv 2025, 02, History-Guided Video Diffusion. [Website](https://boyuan.space/history-guidance/). [Paper](https://arxiv.org/abs/2502.06764).
- arxiv 2025, 01, Inference-Time Alignment in Diffusion Models with Reward-Guided Generation: Tutorial and Review. [Paper](https://arxiv.org/abs/2501.09685).
- arxiv 2024, 05, Bridging Model-Based Optimization and Generative Modeling via Conservative Fine-Tuning of Diffusion Models. [Paper](https://arxiv.org/abs/2405.19673).
- NIPS 2024, Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. [Website](https://boyuan.space/diffusion-forcing/).
- ICML 2022, Learning Iterative Reasoning through Energy Minimization. [Paper](https://arxiv.org/abs/2206.15448).
- ICRA 2023, NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration. [Paper](https://arxiv.org/abs/2310.07896).
- ICML 2024, Video as the New Language for Real-World Decision Making. [Paper](https://arxiv.org/abs/2402.17139).
## related: Generative simulation
- arxiv 2024, 06, RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots. [Paper](https://arxiv.org/abs/2406.02523).
## related: RL in the Real World
- arxiv 2021, 02, NeoRL: A Near Real-World Benchmark for Offline Reinforcement Learning. [Paper](https://arxiv.org/abs/2102.00714). [Website](http://polixir.ai/research/neorl)
