
        My CoRL 2026 Reading List
CoRL 2026 in Austin has 687 accepted papers. Nobody is reading all of them, so I ranked them against my group’s research and kept the top 101: RL fine-tuning of robot policies, world models and world action models, vision-language-action (VLA) models, learning from human video, and imitation learning. I’m sharing the list here in case it’s useful to others at the conference.
Papers are grouped into P0 (must read, 26 papers), P1 (should read, 45), and P2 (nice to read, 30). Each entry has a one-line summary written from the abstract, a link to the paper on OpenReview, any oral or spotlight talk it gives, and its poster: the poster session, the poster number, and the room.
  Orals and spotlights are at the Hilton Austin. Poster sessions are at the JW Marriott: posters 1–116 in Lone Star (3rd floor), posters 117–232 in Griffin (2nd floor).
  11 papers on the list are orals and 13 are spotlights. Monday’s sessions have already happened, but every paper is on OpenReview.
Download the list as BibTeX for Paperpile, Zotero, Mendeley, or any reference manager: all 101 papers  · P0 must read  · P1 should read  · P2 nice to read . Each entry includes the abstract, the OpenReview link, the arXiv ID where there is one, and the session and poster details.
P0 — Must Read
RL Fine-Tuning of VLAs & Generative Policies
  EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models — Dong, Hung, Gao et al. — OpenReview  — Poster Session 1, poster #98: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Stable, sample-efficient real-world RL fine-tuning of pretrained VLAs that reaches deployment-level success on hard manipulation tasks.
  Flow Policy Gradients for Robot Control (Oral) — Yi, Choi, Singh et al. — OpenReview  — Oral O4, Tue 11/10, 9:00–9:30 AM, Hilton Austin · Poster Session 2, poster #63: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Makes flow-matching policy gradients practical for expressive policies, from scratch and fine-tuning, with sim-to-real on two humanoids.
  RL Token: Bootstrapping Online RL with Vision-Language-Action Models (Oral) — Xu, Springenberg, Equi et al. — OpenReview  — Oral O4, Tue 11/10, 9:00–9:30 AM, Hilton Austin · Poster Session 2, poster #64: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Exposes a compact “RL token” from a VLA and trains a small actor-critic on it, anchored to the VLA, for hours-scale real-world RL.
  Scaling Sim-to-Real VLA Reinforcement Learning with Generative 3D Worlds — Choi, Wang, Su et al. — OpenReview  — Poster Session 1, poster #99: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
RL fine-tuning of VLAs in diverse generated 3D simulation scenes, avoiding the scene overfitting of real-world RL, with sim-to-real transfer.
  RoboFFT: Finetuning generative robot policy via online reinforcement learning with forward process — Li, Hu, Wang et al. — OpenReview  — Poster Session 3, poster #66: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Online RL fine-tuning of diffusion and flow policies via forward noising and a weighted flow-matching surrogate for the policy ratio.
RL Algorithms & Exploration
  Learning Object Manipulation from Scratch via Contrastive Interaction — Shen, Chuck, Feng et al. — OpenReview  — Poster Session 1, poster #138: Mon 11/9, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Shows interaction-induced mode changes break contrastive RL in manipulation; interaction-weighted resampling fixes it, learning from scratch.
  Efficient On-policy Visual-RL via Stochastic Decoupled Policy Gradient — You, Liu, Zong et al. — OpenReview  — Poster Session 3, poster #61: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
On-policy visual RL that estimates policy gradients from perturbed rollouts, training visuomotor policies in hours on one GPU, with sim-to-real.
  FlashRFCL: Demonstration-Efficient Sparse Reward RL via Reverse-Forward Curriculum Learning in High-Throughput GPU Simulation — Choi, Tao, Christensen — OpenReview  — Poster Session 1, poster #96: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Sparse-reward RL from a single demonstration using a reverse-forward curriculum and high-throughput GPU simulation.
  Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly? (Oral) — Lum, Kedia, Liu et al. — OpenReview  — Oral O5, Tue 11/10, 10:30–11:00 AM, Hilton Austin · Poster Session 2, poster #126: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Task-agnostic RL play pretraining for a multi-fingered hand, and a study of which play factors matter for precise assembly.
  Learning Loco-Manipulation From SMPC Demonstrations With Sparse Offline-to-Online RL (Spotlight) — Schuck, Sorokin, Manni et al. — OpenReview  — Spotlight S7, Wed 11/11, 9:00–9:30 AM, Hilton Austin · Poster Session 3, poster #201: Wed 11/11, 10:30 AM–12:00 PM, Griffin (2nd floor), JW Marriott
Sampling-based MPC in simulation generates offline data so off-policy RL can learn loco-manipulation from sparse rewards without shaping.
  HARBOR: A Harness Framework for Agentic Robot Reinforcement Learning — Li, Jin, Liu et al. — OpenReview  — Poster Session 1, poster #106: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Agentic harness that automates the robot RL workflow, from environment setup and rewards to tuning and sim-to-real training.
World Models & World Action Models
  SafeDojo: Safe Reinforcement Learning for VLA via Interactive World Model — Tang, Jia, Chu et al. — OpenReview  — Poster Session 3, poster #28: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Model-based safe RL for VLAs inside an interactive video world model, with imagined progress rewards and a learned safety head.
  Large Video Planner Enables Generalizable Robot Control — Chen, Zhang, Geng et al. — OpenReview  — Poster Session 1, poster #57: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Robot foundation model built on internet-scale video pretraining rather than a VLM, planning in video and generalizing to new tasks.
  World Action Models are Zero-shot Policies (Oral) — Ye, Ge, Zheng et al. — OpenReview  — Oral O2, Mon 11/9, 11:00–11:30 AM, Hilton Austin · Poster Session 1, poster #51: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
DreamZero: a world action model on a video diffusion backbone that jointly predicts video and actions, doubling VLA generalization to new tasks.
VLAs & Generalist Policies
  Improving Robotic Generalist Policies via Flow Reversal Steering — Tang, Chen, Wagenmaker et al. — OpenReview  — Poster Session 2, poster #39: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Inverts suboptimal actions to their latent noise and maps them to nearby modes of a flow-matching generalist, steering it to better behavior.
  RoboTTT: Context Scaling for Robot Policies (Oral) — Jiang, Chebotar, Zheng et al. — OpenReview  — Oral O1, Mon 11/9, 10:30–11:00 AM, Hilton Austin · Poster Session 1, poster #47: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Test-time training scales policy context to 8K timesteps without extra latency, enabling in-context imitation from human video and policy improvement.
Learning from Human Video
  LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos — Xiang, Liu, Li et al. — OpenReview  — Poster Session 2, poster #60: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Continuous latent action model on egocentric human video as pretraining supervision for a VLA, then embodiment alignment and post-training.
  ELA: End-to-End Latent Action Policy Learning from Video Demonstrations — Hsu, Stone — OpenReview  — Poster Session 1, poster #16: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Jointly trains the latent action model, latent policy and action decoder so latent actions are optimized for control from video demonstrations.
  EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data (Oral) — Zheng, Niu, Xie et al. — OpenReview  — Oral O5, Tue 11/10, 10:30–11:00 AM, Hilton Austin · Poster Session 2, poster #54: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
VLA trained on 20,854 hours of egocentric human video; finds a log-linear scaling law that predicts real dexterous robot performance.
  EgoEngine: From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations — Liu, Cheng, Yin et al. — OpenReview  — Poster Session 2, poster #57: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Converts egocentric human videos into robot observation videos and executable dexterous action trajectories for policy training.
  Imitation from Heterogeneous Demonstrations using Grounded Latent-Action World Models — Wang, Lei, Watson et al. — OpenReview  — Poster Session 3, poster #30: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Grounds actions in prediction with a latent-action world model so heterogeneous, partly action-free data share one action representation.
Imitation Learning & Data Quality
  Scalable Behavior Cloning with Open Data, Training, and Evaluation — Allshire, Singh, Singh et al. — OpenReview  — Poster Session 1, poster #113: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Fully open behavior-cloning stack: 3,500 hours of bimanual teleop data, hardware, training and correlated sim/real evaluation of DiT and VLA recipes.
Rewards & Progress Models
  Online World Modeling Enables Real-World Inverse Reinforcement Learning from Observation — Han, Nemekhbold, Shen et al. — OpenReview  — Poster Session 3, poster #32: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
First real-world inverse RL from observation only (video, no actions or rewards), learning visual manipulation from scratch with an online world model.
Cross-Embodiment & Robot Design
  Morphology-Conditioned World Model for Cross-Embodiment Quadrupedal Locomotion — Danesh, Li, Abyaneh et al. — OpenReview  — Poster Session 3, poster #156: Wed 11/11, 10:30 AM–12:00 PM, Griffin (2nd floor), JW Marriott
Conditions one world model on scale-invariant morphology features and trains policies inside it, transferring zero-shot to unseen quadrupeds.
  Shape Your Body: Value Gradients for Multi-Embodiment Robot Design (Spotlight) — Bohlinger, Peters — OpenReview  — Spotlight S5, Tue 11/10, 2:30–3:00 PM, Hilton Austin · Poster Session 2, poster #115: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Trains one multi-embodiment value function, then uses its gradients as a differentiable surrogate to optimize robot designs without new RL loops.
  MorFiC: Fixing Value Miscalibration for Zero-Shot Quadruped Transfer — Mishra, Raj, Xiao et al. — OpenReview  — Poster Session 3, poster #155: Wed 11/11, 10:30 AM–12:00 PM, Griffin (2nd floor), JW Marriott
A shared critic averages incompatible value targets across quadruped morphologies; fixing this miscalibration enables zero-shot transfer.
P1 — Should Read
RL Fine-Tuning of VLAs & Generative Policies
  Training Robots to Reason in Natural Language via Reinforcement Learning — Wu, Qu, Hu et al. — OpenReview  — Poster Session 1, poster #101: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Trains VLMs to reason in natural language to guide low-level manipulation policies: reasoning-trace mid-training, then RL post-training.
  Optimal Transport Q-Learning for Flow Policy Steering and Acceleration — Sochopoulos, Whitammer, Tsagkas et al. — OpenReview  — Poster Session 3, poster #67: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Advantage-weighted conditional optimal-transport flow matching to RL fine-tune and accelerate suboptimal flow policies.
  Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies — Wang, Li, Xie et al. — OpenReview  — Poster Session 1, poster #85: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Fleet-scale offline-to-online RL for continual post-training of generalist VLAs from autonomous rollouts and human interventions.
  X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching — Yang, Zeng, Cai et al. — OpenReview  — Poster Session 1, poster #174: Mon 11/9, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
RL post-training of navigation diffusion policies with group Q-score reweighted matching, generalizing to new embodiments and behaviors.
RL Algorithms & Exploration
  A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control — Zhang, Castro, Yin et al. — OpenReview  — Poster Session 1, poster #105: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Success-guided sampling of reset configurations keeps mega-scale parallel RL on the frontier of capability, so returns keep scaling.
  Combining Constrained Sampling and Reinforcement Learning for Manipulation (Spotlight) — Toussaint, Braun, Jordana et al. — OpenReview  — Spotlight S4, Tue 11/10, 2:00–2:30 PM, Hilton Austin · Poster Session 2, poster #167: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Constrained-manifold state sampling for resets boosts exploration in goal-conditioned RL for contact-rich non-prehensile manipulation.
  Emergent Control via Prior-Based Free Energy Minimization (Spotlight) — Kolachalam, Malhotra — OpenReview  — Spotlight S6, Wed 11/11, 8:30–9:00 AM, Hilton Austin · Poster Session 3, poster #72: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Control by minimizing deviation from priors over preferred states (free energy) instead of maximizing reward, model-based and model-free.
World Models & World Action Models
  Trust-Calibrated VLA Policy Post-Training with Evolving Imagination — Li, Qin, Zhou et al. — OpenReview  — Poster Session 1, poster #35: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
VLA RL post-training with an evolving world model and reward head, using imagined rollouts in proportion to how much they can be trusted.
  Chunk-wise Model-based Policy Optimization (Spotlight) — Song, Wang, Li — OpenReview  — Spotlight S2, Mon 11/9, 2:30–3:00 PM, Hilton Austin · Poster Session 1, poster #109: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Model-based RL with a chunk world model as a differentiable evaluator for chunked actors, avoiding recurrent rollouts and BPTT.
  Training Controllable World Models via Action Influence Maximization — Bateman, Yin, Zheng et al. — OpenReview  — Poster Session 3, poster #37: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
World models trained on successful trajectories ignore actions; an action-influence (empowerment) objective restores controllability.
  WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL — Jiang, Zhou, Jiang et al. — OpenReview  — Poster Session 3, poster #25: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Makes world-model RL for VLAs reliable by regulating how the policy uses imperfect imagined rollouts.
  Coupled Local and Global World Models for Efficient First Order RL — Amigo, Khorrambakht, Mansard et al. — OpenReview  — Poster Session 3, poster #27: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
First-order RL with a large world model for forward rollouts and a lightweight latent surrogate for gradients, on real robots.
  PlayWorld: Learning Robot World Models from Autonomous Play Data — Yin, Mei, Zheng et al. — OpenReview  — Poster Session 3, poster #26: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Trains robot world models on autonomous play data, reducing contact hallucinations compared with demonstrations or policy rollouts.
  Hierarchical Planning with Latent World Models — Zhang, Terver, Zholus et al. — OpenReview  — Poster Session 3, poster #1: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Hierarchical MPC on visual latent world models learned at multiple temporal scales for long-horizon zero-shot control.
  Light-WAM: Efficient World Action Models with State-Fusion Action Decoding — Li, Cheng, Wang et al. — OpenReview  — Poster Session 1, poster #55: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Lightweight world action model with latent future-video supervision and a state-fusion action expert for efficient closed-loop control.
  PLaW-VLA: Predictive Latent World Modeling for Vision-Language-Action Policies — Liu, Guo, Huang et al. — OpenReview  — Poster Session 3, poster #2: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
VLA that predicts future world states in a pretrained latent space instead of pixels, inside one Mixture-of-Transformers.
VLAs & Generalist Policies
  ELASTIC: Efficiently Learning to Adaptively Scale Test-Time Compute for Generative Control Policies — Li, Swamy, Bisk et al. — OpenReview  — Poster Session 2, poster #3: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Learns to allocate test-time compute between more denoising steps and more parallel samples for diffusion and flow VLA policies, per state.
  MAESTRO: Orchestrating Robotics Modules with Coding Agents for Zero-Shot Generalist Robots — Shi, Yang, Chao et al. — OpenReview  — Poster Session 1, poster #203: Mon 11/9, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
A VLM coding agent orchestrating perception, planning and control tools matches VLAs zero-shot, without new teleop data.
  OmniGuide: Universal Spatial Guidance Fields for Enhancing Generalist Robot Policies — Song, Le, Park et al. — OpenReview  — Poster Session 1, poster #32: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Expresses 3D, semantic and human-pose guidance as differentiable energy fields that steer VLA action sampling at test time.
  PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations (Spotlight) — Zhang, Zhao, Fan et al. — OpenReview  — Spotlight S1, Mon 11/9, 9:00–9:30 AM, Hilton Austin · Poster Session 1, poster #14: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
VLA pretraining as goal-conditioned contrastive RL: language goals and state-actions share an embedding approximating goal occupancy.
  FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation — Yuan, Zhang, Zhou et al. — OpenReview  — Poster Session 2, poster #142: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Generalist tactile policy pretrained on 3,000 hours from 21 sensors, with morphology-aware tokens for cross-sensor transfer.
  Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization (Oral) — Luo, Wang, Zhang et al. — OpenReview  — Oral O1, Mon 11/9, 10:30–11:00 AM, Hilton Austin · Poster Session 1, poster #67: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Human-centric cross-embodiment VLA pretrained on 35,000 hours from humans and 30 robots, with a unified action space.
  MolmoAct2: Action Reasoning Models for Real-World Deployment (Spotlight) — Fang, Duan, Clay et al. — OpenReview  — Spotlight S1, Mon 11/9, 9:00–9:30 AM, Hilton Austin · Poster Session 1, poster #28: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Fully open action-reasoning VLA built for deployment: embodied-reasoning VLM backbone, new datasets and low-cost platforms.
  π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities (Oral) — Ai, Amin, Aniceto et al. — OpenReview  — Oral O1, Mon 11/9, 10:30–11:00 AM, Hilton Austin · Poster Session 1, poster #63: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
π0.7: diverse context conditioning (strategy, metadata, subgoal images) makes a generalist steerable and lets it use suboptimal data.
Learning from Human Video
  Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data (Spotlight) — Wang, Lin, Chen et al. — OpenReview  — Spotlight S3, Tue 11/10, 8:30–9:00 AM, Hilton Austin · Poster Session 2, poster #55: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Converts egocentric human video into 18,561 hours of robot data across 15 morphologies for VLA pretraining at scale.
  Monkey See, Can Monkey Do? A Benchmark for Evaluating Robot Skill Learning by Observation — Gu, Gupta, Sah et al. — OpenReview  — Poster Session 2, poster #106: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
RoboReel: a unified real-world benchmark for policies that learn from human videos.
  Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation — Wang, Wu, Abbeel et al. — OpenReview  — Poster Session 2, poster #184: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Amplifies a few human videos into many counterfactual interaction videos, retargeted through real-to-sim, to train humanoid loco-manipulation.
  WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time — Feng, Han, Lyu et al. — OpenReview  — Poster Session 3, poster #7: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Test-time training of a world-action model on human egocentric play to handle out-of-distribution object configurations.
  LUCID: Learning Embodiment-Agnostic Intent Models from Unstructured Human Videos for Scalable Dexterous Robot Skill Acquisition — Gupta, Shi, Yuan — OpenReview  — Poster Session 3, poster #54: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Learns embodiment-agnostic intent from unstructured internet human videos, executed by a tracking policy trained in simulation.
  ACE-Ego: Unifying Egocentric Human and Robotic Data for VLA Pretraining — Li, Zhao, Liu et al. — OpenReview  — Poster Session 2, poster #61: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
VLA pretraining on robot data plus pseudo-labeled egocentric human video, unified through camera-space actions and morphology tokens.
  HARP-VLA: Human-Robot Aligned Representation Learning for Vision-Language-Action Model — Zhu, Yuan, Liu et al. — OpenReview  — Poster Session 1, poster #17: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Aligns human and robot visual representations using a few paired demonstrations so latent actions from human video transfer to VLAs.
  HuRo: Robotizing Human Videos for Scalable VLA Pretraining — Jeong, Joo, Kang et al. — OpenReview  — Poster Session 2, poster #59: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Robotizes heterogeneous human videos (visuals and action trajectories) as a scalable source for VLA pretraining.
  The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction — Zhou, Cai, Wang et al. — OpenReview  — Poster Session 2, poster #88: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Four-level benchmark testing whether robots imitate the intent of a human demonstration rather than replaying its trajectory.
Imitation Learning & Data Quality
  Q-OIL: Optimistic Exploration in Learning from Human Interventions — Teegavarapu, Poddar, Qi et al. — OpenReview  — Poster Session 1, poster #103: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Real people intervene when progress stalls and correct suboptimally; a progress-based intervention model plus optimistic exploration.
  Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics — Wei, Pfaff, Cohn et al. — OpenReview  — Poster Session 3, poster #77: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Imitation from suboptimal data by restricting low-quality demonstrations to high and low diffusion noise levels.
Rewards & Progress Models
  WARP-RM: A Warp-Augmented Relative Progress Reward Model for Data Curation — Yu, Goldberg, Kondap et al. — OpenReview  — Poster Session 3, poster #68: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Self-supervised relative-progress reward model from time-warped demonstrations, used to curate and weight mixed-quality teleop data.
Sim-to-Real, Dexterity & Data Generation
  Dex4D: Task-Agnostic Point Track Policy for Zero-Shot Dexterous Manipulation — Kuang, Park, Fragkiadaki et al. — OpenReview  — Poster Session 3, poster #118: Wed 11/11, 10:30 AM–12:00 PM, Griffin (2nd floor), JW Marriott
Anypose-to-anypose dexterous point-track policy trained in simulation on thousands of objects, recomposed zero-shot for real tasks.
  ExpertGen: Scalable Sim-to-Real Expert Policy Learning from Imperfect Behavior Priors — Xu, Gong, Minniti et al. — OpenReview  — Poster Session 3, poster #85: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Steers a diffusion behavior prior with RL over its initial noise in parallel simulation, producing expert data without human demos.
  Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement — Kim, Saito, Kim et al. — OpenReview  — Poster Session 3, poster #65: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Object-pose residual RL trained in simulation improves a frozen VLA’s precision zero-shot on the real robot.
  TacCoRL: Integrating Tactile Feedback into VLA via Simulation — Ma, Liang, Yu et al. — OpenReview  — Poster Session 2, poster #149: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Adds touch to VLAs through sim-real co-training and simulation RL focused on near-failure contact states.
  Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration — Zhu, Liu, Jain et al. — OpenReview  — Poster Session 3, poster #47: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Object-centric contact-wrench guidance from human demonstrations makes dexterous RL scale, with a 4,739-task bimanual benchmark.
Planning & Compositional Generalization
  What Do You Need for Compositional Generalization in Diffusion Planning? — Clark, Shkurti — OpenReview  — Poster Session 2, poster #20: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Shift equivariance and local receptive fields explain when diffusion planners trained by BC stitch and compose sub-trajectories.
  Long-Horizon Multi-Object Rearrangement via Distilled Tree Search over Continuous Actions (Oral) — Saha, Hong, Li et al. — OpenReview  — Oral O6, Tue 11/10, 11:00–11:30 AM, Hilton Austin · Poster Session 2, poster #176: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Distills MCTS over continuous actions in GPU-parallel simulation into policies for long-horizon multi-object rearrangement.
  Leave It Better Than Found: Value-Guided Cooperation in Sequential Tasks (Spotlight) — Andreu, Amir, Marino et al. — OpenReview  — Spotlight S3, Tue 11/10, 8:30–9:00 AM, Hilton Austin · Poster Session 2, poster #171: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Value-guided ad hoc cooperation: a robot leaves the workspace in a state the next, unknown robot can continue from.
  Recover, Discover, Plan: Learning Skills and Concepts from Robot Failures — Li, Mishra, Liu et al. — OpenReview  — Poster Session 2, poster #94: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Learns recovery skills with RL and discovers relational predicates from failure-recovery experience to support abstract planning.
P2 — Nice to Read
RL Fine-Tuning of VLAs & Generative Policies
  Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models — Shi, Chen, Gao et al. — OpenReview  — Poster Session 3, poster #64: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Sim-real co-training with RL in simulation plus a supervised loss on real data, instead of SFT-only co-training.
  CounterAlign: Counterfactual Supervision for Vision-Language-Action Models — Kondoh, Ota, Kanezaki et al. — OpenReview  — Poster Session 2, poster #31: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Offline RL for VLAs from expert data alone, using instruction relabeling to create counterfactual negative supervision.
RL Algorithms & Exploration
  Tactile Curiosity Drives Robot Interaction — Iten, Proshkin, Sukhija et al. — OpenReview  — Poster Session 3, poster #148: Wed 11/11, 10:30 AM–12:00 PM, Griffin (2nd floor), JW Marriott
Uses touch as the signal for uncertainty-driven exploration so RL agents explore contacts rather than free space.
  ENPIRE: Agentic Robot Policy Self-Improvement in the Real World — Xiao, Xie, Zhang et al. — OpenReview  — Poster Session 1, poster #202: Mon 11/9, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Harness for coding agents that runs real-world policy self-improvement loops: reset, execute, verify, refine.
World Models & World Action Models
  Scaling World Model for Hierarchical Manipulation Policies — Qian, Wang, Song et al. — OpenReview  — Poster Session 3, poster #17: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Hierarchical VLA: a pretrained world model generates subgoal images and a VLA executes them, improving OOD generalization.
  FACT: Failure-Aware Causal Training for World-Action Models — Peng, Liang, Yan et al. — OpenReview  — Poster Session 3, poster #4: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Action-conditioned world-action model that learns from failure rollouts, so it predicts the consequences of bad actions.
  SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model (Oral) — Liao, Shao, Long et al. — OpenReview  — Oral O7, Tue 11/10, 11:30 AM–12:00 PM, Hilton Austin · Poster Session 2, poster #217: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
One differentiable world model of shared dynamics with analytic gradients trains a humanoid whole-body tracker in 3 hours on one GPU.
  World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis — Yang, Liu, Kou et al. — OpenReview  — Poster Session 1, poster #56: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Autoregressive world-language-action model jointly predicting text subtasks, subgoal images and actions.
  W2A: Distilling World Priors into Action Models for Real-Time Robotic Control — Zhang, Yongjin, Weng et al. — OpenReview  — Poster Session 2, poster #4: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Distills video-diffusion features into an action model during training so no video is generated at test time.
  Targeting World Models to Compromise Robot Learning Pipelines (Spotlight) — Rathbun, Agha, Mahmud et al. — OpenReview  — Spotlight S2, Mon 11/9, 2:30–3:00 PM, Hilton Austin · Poster Session 1, poster #54: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Shows learned world models in robot learning pipelines open a stealthy attack surface that can compromise downstream policies.
  TAPESTRY: A Single Backbone for Video Generation and Robot Control — Collins, Lee, Wachakorn et al. — OpenReview  — Poster Session 3, poster #6: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Repurposes a pretrained video diffusion transformer as a policy by treating actions as single video-patch tokens.
  MDF++: Visuo-Tactile World Action Model with End-to-end Representation Learning — Huang, Yu, Charlick et al. — OpenReview  — Poster Session 2, poster #158: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Visuo-tactile world-action model that learns control-relevant representations end-to-end, avoiding feature collapse.
VLAs & Generalist Policies
  APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies — Xu, Zhu, Chen et al. — OpenReview  — Poster Session 2, poster #21: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Pretrains the VLA action expert as a language-agnostic vision-action prior to stop visual shortcuts and improve instruction generalization.
  SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation (Spotlight) — Kim, Pak, Long et al. — OpenReview  — Spotlight S5, Tue 11/10, 2:30–3:00 PM, Hilton Austin · Poster Session 2, poster #173: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Conditions a VLA on an online spatiotemporal neural-point map of the scene and the robot body for long-horizon mobile manipulation.
  Learning What to Say to Your VLA: Mostly Harmless Vision Language Action Model Steering (Spotlight) — Jeong, Swamy, Bajcsy — OpenReview  — Spotlight S1, Mon 11/9, 9:00–9:30 AM, Hilton Austin · Poster Session 1, poster #64: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Searches for language that improves a frozen VLA’s closed-loop success and distills it into a test-time language feedback policy.
Learning from Human Video
  EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data — Li, Yin, Lin et al. — OpenReview  — Poster Session 2, poster #62: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Controlled study of which world-prediction target makes world action models best at transferring in-the-wild egocentric human data.
Imitation Learning & Data Quality
  FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space (Spotlight) — Murray, Chen, Bagaria et al. — OpenReview  — Spotlight S3, Tue 11/10, 8:30–9:00 AM, Hilton Austin · Poster Session 2, poster #29: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Adapts frozen flow and diffusion policies from human interventions by inverting expert actions to latent noise.
  SynthICL: Scalable In-context Imitation Learning with Synthetic Data — Qian, Fan, Ren et al. — OpenReview  — Poster Session 3, poster #79: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
In-context imitation policies trained entirely on RGB-only synthetic data, with subgoal prediction.
  Graph-Reweighted Behavior Cloning with Human Interventions for Precise Robotic Manipulation — Li, Yang, Hu et al. — OpenReview  — Poster Session 2, poster #65: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Weights action chunks by how much they reduce remaining steps on a goal-reachability graph, for precise VLA post-training.
  Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets (Oral) — Sirigiri, Lara, Agia et al. — OpenReview  — Oral O3, Mon 11/9, 11:30 AM–12:00 PM, Hilton Austin · Poster Session 1, poster #199: Mon 11/9, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Signature-kernel entropy measures trajectory dataset diversity and gives a fast, model-free way to curate demonstrations.
  Progress-Guided Preference Generation for Aligning Generative Control Policies — Luu, Ton, Nguyen et al. — OpenReview  — Poster Session 2, poster #30: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Unsupervised preference labels from progress relative to a few expert demos fine-tune generative policies without human annotation.
  MIRROR: Reflecting Hidden Symmetries from Offline Robot Interaction Data — Liu, Yao, Huang et al. — OpenReview  — Poster Session 3, poster #116: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Discovers how symmetry groups act on pixel observations and actions from offline data, for symmetry-aware policy learning.
Rewards & Progress Models
  HiRE: Hindsight Reward Editing for Policy Finetuning — Niu, Han, Ji et al. — OpenReview  — Poster Session 1, poster #97: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Training-free reward editing that calibrates foundation-model rewards by contrasting successful and failed trajectories in hindsight.
  TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics — Chen, Harrison, Lee et al. — OpenReview  — Poster Session 1, poster #94: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Zero-shot dense progress rewards from a video-language model’s token probabilities, without reward-model training.
  RARM: Confidence-Gated Progress Reward Modeling for RL in Manipulation — Yang, Wang, Jing et al. — OpenReview  — Poster Session 1, poster #95: Mon 11/9, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Reference-anchored, confidence-gated progress reward model from a single demonstration, robust to failures and reward hacking.
Sim-to-Real, Dexterity & Data Generation
  HumanoidMimicGen Data Generation for Loco-Manipulation via Whole-Body Planning — Lin, Mandlekar, Garrett et al. — OpenReview  — Poster Session 2, poster #185: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
MimicGen-style data generation for humanoid loco-manipulation, adapting whole-body skills from a few demos with planning.
  Assembling Two Parts in One Hand — Pei, Wu, Zhang et al. — OpenReview  — Poster Session 3, poster #121: Wed 11/11, 10:30 AM–12:00 PM, Griffin (2nd floor), JW Marriott
RL for in-hand assembly of two parts in a single dexterous hand, with function-based rewards and sim-to-real.
  Building Scalable Real-World Robot Data Generation via Compositional Simulation — Qin, Ma, Kang et al. — OpenReview  — Poster Session 3, poster #86: Wed 11/11, 10:30 AM–12:00 PM, Lone Star (3rd floor), JW Marriott
Combines classical and neural simulation in a real-sim-real loop to generate large, real-consistent action-video datasets.
  Dex-X: Learning Visual-Tactile Dexterous Manipulation From Human Demonstration with Simulated Interaction — Chen, Ruan, Cao et al. — OpenReview  — Poster Session 2, poster #151: Tue 11/10, 5:00–6:30 PM, Griffin (2nd floor), JW Marriott
Recovers tactile signals for human video demonstrations by reconstructing them in simulation, then learns visual-tactile dexterous policies.
Cross-Embodiment & Robot Design
  Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design — Ha, Liu, Song — OpenReview  — Poster Session 2, poster #116: Tue 11/10, 5:00–6:30 PM, Lone Star (3rd floor), JW Marriott
Diffusion transformer over unified embodiment, state and action tokens that generates robot designs for target motions.
    Workshops
CoRL 2026 has 32 workshops and tutorials, all on Thursday, Nov 12, at the JW Marriott and the Hilton Austin. These are the ones closest to RL, world models, learning from human video, and adapting pretrained policies.
Morning (8:30 AM–12:30 PM):
  R2RL: Roadmap to Sample-Efficient Real-World Reinforcement Learning — JW Marriott, Room 213 — workshop page 
  Continually Self-Improving Robots (CSIR-2026) — JW Marriott, Room 202 — workshop page 
  Learn@Deploy: Robot Learning at Deployment — JW Marriott, Room 205 — workshop page 
  Learning from Corrections and Interventions — JW Marriott, Room 305 — workshop page 
  Do Robots Need World Models? A Debate on Dynamics Learning, Planning, and End-to-End Robot Learning — JW Marriott, Brazos — workshop page 
  Scaling H2R: Scaling Laws and Diversity in Human-to-Robot — JW Marriott, Rooms 301–302 — workshop page 
  Compositional Generalization for Real-World Robot Learning — JW Marriott, Rooms 310–311 — workshop page 
  From GPU-Accelerated Simulation to Scalable and Generalizable Real Robot Policy Learning — JW Marriott, Room 212 — workshop page 
Afternoon (2:00–6:00 PM):
  Pretrain to Adapt: What Makes a Pretrained Policy Adaptable? — JW Marriott, Rooms 301–302 — workshop page 
  Modeling Uncertainty in Robotic World Models — JW Marriott, Rooms 310–311 — workshop page 
  Bringing Physics Simulation and World Models Together for Robotic Manipulation — JW Marriott, Brazos — workshop page 
  The Action Gap: Large Generative Models (Video, Image, 3D) Learn Rich Knowledge — JW Marriott, Room 208 — workshop page 
  “Oops, I Erred”: Collecting, Curating and Using Imperfect Data for Realistic Scenarios — JW Marriott, Room 305 — workshop page 
  Sim-to-Real-to-Field (SRF): The Science of Transfer — Hilton, Salon E — workshop page 
  From Human Videos to Dexterous Robot Capability: A Hands-on Tutorial — Hilton, Salon D — tutorial page 
  Workshop on Agentic Robotics — JW Marriott, Rooms 203–204 — workshop page 
How the workshops connect to the papers:
  R2RL, CSIR, Learn@Deploy and Pretrain to Adapt go with the RL fine-tuning sections above.
  Do Robots Need World Models?, Modeling Uncertainty in Robotic World Models, Physics Simulation and World Models, and The Action Gap go with the world model sections.
  Scaling H2R and the human-video tutorial pair with the learning-from-human-video papers.
  Learning from Corrections and Interventions and “Oops, I Erred” pair with the imitation learning and data-quality papers.
  (See the full workshop schedule  for everything else on offer.)
Themes Worth Following
A few threads run through this year’s program:
  RL fine-tuning of VLAs and generative policies. This is the robotics side of the RLVR/GRPO work on LLMs. Start with RL Token, EXPO-FT, and Flow Policy Gradients, then see Learning While Deploying for fleet-scale RL and Training Robots to Reason in Natural Language for RL on a VLM reasoner.
  World action models. Policies built on video diffusion backbones that predict the future along with actions. Start with DreamZero (World Action Models are Zero-shot Policies), then TAPESTRY, W2A, and FACT, which teaches the model what bad actions do.
  World models as simulators for RL. SafeDojo, WoVR, and MERL each try to make RL on imagined rollouts trustworthy, and Training Controllable World Models shows why world models trained on successes ignore actions.
  Human video at scale. EgoScale finds a log-linear scaling law for egocentric human data, and Ego2Robot and Being-H0.5 turn tens of thousands of hours of human data into VLA pretraining. LAPT-Scale and ELA learn latent actions from video instead.
  Progress rewards. WARP-RM, TOPReward, and RARM all produce dense task-progress signals without hand-designed rewards.
How I Built This List
I ranked the papers with a model of my research interests, then grouped and summarized the top of the ranking.
  I started from all 687 accepted papers, with their titles and abstracts.
  The model is trained only on my side: my publications and the papers I added to my library in the last six months. These are contrasted with a background of recent arXiv papers and the CoRL program itself, so each paper is compared with a typical CoRL paper. It combines an arXiv Sanity -style linear SVM with TPMS -style affinity, the classic reviewer-matching score.
  The top 101 papers became the list, split by rank into P0, P1, and P2. I then sorted them into topics, and the one-line summaries were drafted with an LLM from the abstracts.
Every paper on the list links to its OpenReview page, which has the paper PDF. If you’re at CoRL 2026 in Austin and this overlaps with your interests, I hope it saves you some filtering time.
Sources:
  CoRL 2026 accepted papers : the official program, and the source of the titles, OpenReview links, oral and spotlight sessions, and poster sessions and numbers.
  CoRL 2026 papers : a community-maintained list with the authors and abstracts of every accepted paper, matched to the program by OpenReview ID.
  CoRL 2026 workshop schedule : the workshop times and rooms.
A note on selection: this list is based on my own research interests, not on the quality of the work. There are great papers across all of CoRL 2026, and many excellent ones aren’t here simply because they fall outside my group’s areas.
