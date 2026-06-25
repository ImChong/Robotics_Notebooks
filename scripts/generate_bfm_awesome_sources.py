#!/usr/bin/env python3
"""Generate sources/papers/bfm_awesome_* archives for awesome-bfm-papers 41 papers + 10 datasets."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "sources" / "papers"
TODAY = date.today().isoformat()
CATALOG = "bfm_awesome_41_catalog.md"
WECHAT = "wechat_embodied_ai_lab_bfm_41_papers_survey.md"

# (id, slug, title, year, venue, paper_url, code_url, group, wechat_note, wiki_hints, skip_if_exists)
PAPERS: list[dict] = [
    {
        "id": 1,
        "slug": "bfm_zero_arxiv_2511_04131",
        "title": "BFM-Zero: Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2511.04131",
        "code": "https://github.com/LeCAR-Lab/BFM-Zero",
        "group": "forward-backward",
        "note": "latent prompt 统一目标姿态、奖励优化、恢复与少样本适配；技能切换≈搜索身体潜空间，接近产业「运控基座」。",
        "wiki": [
            "wiki/entities/paper-behavior-foundation-model-humanoid.md",
            "wiki/concepts/behavior-foundation-model.md",
            "wiki/overview/bfm-41-papers-technology-map.md",
        ],
    },
    {
        "id": 2,
        "slug": "metamotivo_arxiv_2504_11054",
        "title": "Zero-shot Whole-body Humanoid Control via Behavioral Foundation Models",
        "year": 2025,
        "venue": "ICLR",
        "paper": "https://arxiv.org/abs/2504.11054",
        "code": "https://github.com/facebookresearch/metamotivo",
        "group": "forward-backward",
        "note": "与 BFM-Zero 对照：zero-shot WBC 依赖可调用行为表示，任务变时尽量在潜空间找方向而非重训。",
        "wiki": [
            "wiki/concepts/behavior-foundation-model.md",
            "wiki/overview/bfm-41-papers-technology-map.md",
        ],
    },
    {
        "id": 3,
        "slug": "fb_aw_arxiv_2412_04368",
        "title": "Finer Behavioral Foundation Models via Auto-regressive Features and Advantage Weighting",
        "year": 2024,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2412.04368",
        "code": "",
        "group": "forward-backward",
        "note": "潜空间要「细」才可被上层精确调用；FB-AW / FB-AWARE 在连续控制 benchmark 上优于粗表征。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 4,
        "slug": "fast_imitation_bfm_neurips_2024",
        "title": "Fast Imitation via Behavior Foundation Models",
        "year": 2024,
        "venue": "NeurIPS",
        "paper": "https://openreview.net/pdf?id=qnWtw3l0jb",
        "code": "",
        "group": "forward-backward",
        "note": "有行为基座后新动作应少走弯路；降低技能扩展的真机数据与训练成本。",
        "wiki": [
            "wiki/methods/imitation-learning.md",
            "wiki/concepts/behavior-foundation-model.md",
        ],
    },
    {
        "id": 5,
        "slug": "learning_one_representation_neurips_2021",
        "title": "Learning One Representation to Optimize All Rewards",
        "year": 2021,
        "venue": "NeurIPS",
        "paper": "https://proceedings.neurips.cc/paper_files/paper/2021/file/003dd617c12d444ff9c80f717c3fa982-Paper.pdf",
        "code": "https://github.com/ahmed-touati/controllable_agent",
        "group": "forward-backward",
        "note": "FB 嵌入统一表示，面对不同 reward 推导策略；为 BFM「先学可迁移结构」打底。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 6,
        "slug": "successor_states_arxiv_2101_07123",
        "title": "Learning Successor States and Goal-Dependent Values: A Mathematical Viewpoint",
        "year": 2021,
        "venue": "arXiv",
        "paper": "https://arxiv.org/pdf/2101.07123",
        "code": "",
        "group": "forward-backward",
        "note": "未来状态分布作为可复用控制表示的数学底座。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 7,
        "slug": "sonic_arxiv_2511_07820",
        "title": "Sonic: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2511.07820",
        "code": "https://nvlabs.github.io/SONIC/",
        "group": "goal-conditioned",
        "note": "supersizing motion tracking；运控基座被上层调用前底层动作覆盖面必须足够宽。",
        "wiki": [
            "wiki/methods/sonic-motion-tracking.md",
            "wiki/overview/bfm-41-papers-technology-map.md",
        ],
    },
    {
        "id": 8,
        "slug": "opentrack_arxiv_2509_13833",
        "title": "Track Any Motions under Any Disturbances",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2509.13833",
        "code": "https://github.com/GalaxyGeneralRobotics/OpenTrack",
        "group": "goal-conditioned",
        "note": "跟踪 + 抗扰一体；被推撞后仍能接住才算可用身体能力。",
        "wiki": ["wiki/methods/any2track.md", "wiki/overview/bfm-41-papers-technology-map.md"],
    },
    {
        "id": 9,
        "slug": "ams_arxiv_2511_17373",
        "title": "Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2511.17373",
        "code": "https://github.com/OpenDriveLab/AMS",
        "group": "goal-conditioned",
        "note": "异构数据下敏捷与稳定权衡；MoCap/仿真/视频等混合是 BFM 数据常态。",
        "wiki": ["wiki/methods/ams.md"],
    },
    {
        "id": 10,
        "slug": "twist2_arxiv_2511_02832",
        "title": "TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2511.02832",
        "code": "https://github.com/amazon-far/TWIST2",
        "group": "goal-conditioned",
        "note": "可扩展全身数据采集系统；BFM 进化依赖持续数据管线而非一次性动作集。",
        "wiki": ["wiki/tasks/teleoperation.md"],
    },
    {
        "id": 11,
        "slug": "twist_corl_2025",
        "title": "TWIST: Teleoperated Whole-Body Imitation System",
        "year": 2025,
        "venue": "CoRL",
        "paper": "https://arxiv.org/abs/2505.02833",
        "code": "https://github.com/YanjieZe/TWIST",
        "group": "goal-conditioned",
        "note": "遥操作同时是 BFM 数据生产方式：稳定全身轨迹沉淀为训练语料。",
        "wiki": ["wiki/tasks/teleoperation.md"],
    },
    {
        "id": 12,
        "slug": "clone_corl_2025",
        "title": "CLONE: Closed-Loop Whole-Body Humanoid Teleoperation for Long-Horizon Tasks",
        "year": 2025,
        "venue": "CoRL",
        "paper": "https://proceedings.mlr.press/v305/li25h.html",
        "code": "https://github.com/humanoid-clone/CLONE/",
        "group": "goal-conditioned",
        "note": "长时程闭环遥操作；片段演示不足以支撑复杂任务级 BFM 数据。",
        "wiki": ["wiki/tasks/teleoperation.md"],
    },
    {
        "id": 13,
        "slug": "bfm_humanoid_arxiv_2509_13780",
        "title": "Behavior Foundation Model for Humanoid Robots",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2509.13780",
        "code": "https://bfm4humanoid.github.io/",
        "group": "goal-conditioned",
        "note": "CVAE + 掩码蒸馏多接口 WBC；「人形 BFM」从术语走向系统设计。",
        "wiki": ["wiki/entities/paper-behavior-foundation-model-humanoid.md"],
        "skip": True,
    },
    {
        "id": 14,
        "slug": "hover_arxiv_2410_21229",
        "title": "HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots",
        "year": 2025,
        "venue": "ICRA",
        "paper": "https://arxiv.org/abs/2410.21229",
        "code": "https://github.com/NVlabs/HOVER/",
        "group": "goal-conditioned",
        "note": "统一头/手/身体/根目标的神经全身接口，供上层规划器调用。",
        "wiki": ["wiki/concepts/whole-body-control.md"],
    },
    {
        "id": 15,
        "slug": "intermimic_arxiv_2502_20390",
        "title": "InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions",
        "year": 2025,
        "venue": "CVPR",
        "paper": "https://arxiv.org/abs/2502.20390",
        "code": "https://github.com/Sirui-Xu/InterMimic",
        "group": "goal-conditioned",
        "note": "从纯身体动作走向人-物交互与接触结果。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 16,
        "slug": "modskill_arxiv_2502_14140",
        "title": "ModSkill: Physical Character Skill Modularization",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2502.14140",
        "code": "",
        "group": "goal-conditioned",
        "note": "技能模块化、可组合，避免基座内部技能缠成一团黑箱。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 17,
        "slug": "maskedmimic_tog_2024",
        "title": "MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting",
        "year": 2024,
        "venue": "TOG",
        "paper": "https://research.nvidia.com/labs/par/maskedmimic/assets/SIGGRAPHAsia2024_MaskedMimic.pdf",
        "code": "https://github.com/NVlabs/ProtoMotions",
        "group": "goal-conditioned",
        "note": "稀疏/遮蔽条件下补全全身轨迹；贴近语言只给部分约束的现实。",
        "wiki": ["wiki/entities/protomotions.md"],
    },
    {
        "id": 18,
        "slug": "hgap_arxiv_2312_02682",
        "title": "H-GAP: Humanoid Control with a Generalist Planner",
        "year": 2024,
        "venue": "ICLR",
        "paper": "https://arxiv.org/abs/2312.02682",
        "code": "https://github.com/facebookresearch/hgap",
        "group": "goal-conditioned",
        "note": "generalist planner 管理低层人形控制；BFM 进入系统后的分层问题。",
        "wiki": ["wiki/overview/bfm-41-papers-technology-map.md"],
    },
    {
        "id": 19,
        "slug": "calm_siggraph_2024",
        "title": "CALM: Conditional Adversarial Latent Models for Directable Virtual Characters",
        "year": 2024,
        "venue": "SIGGRAPH",
        "paper": "https://research.nvidia.com/labs/par/calm/assets/SIGGRAPH2023_CALM.pdf",
        "code": "https://github.com/NVlabs/CALM",
        "group": "goal-conditioned",
        "note": "可指挥 latent skill；BFM 前史：技能空间先于大基座。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 20,
        "slug": "moconvq_tog_2023",
        "title": "MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations",
        "year": 2023,
        "venue": "TOG",
        "paper": "https://arxiv.org/pdf/2310.10198",
        "code": "",
        "group": "goal-conditioned",
        "note": "离散动作 token 便于规划器/语言模块组织连续轨迹。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 21,
        "slug": "case_arxiv_2309_11351",
        "title": "CASE: Learning Conditional Adversarial Skill Embeddings for Physics-Based Characters",
        "year": 2023,
        "venue": "SIGGRAPH Asia",
        "paper": "https://arxiv.org/abs/2309.11351",
        "code": "https://github.com/Frank-ZY-Dou/CASE",
        "group": "goal-conditioned",
        "note": "条件技能嵌入；与「按 prompt 调身体」同思路。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 22,
        "slug": "phc_arxiv_2305_06456",
        "title": "PHC: Perpetual Humanoid Control for Real-Time Simulated Avatars",
        "year": 2023,
        "venue": "ICCV",
        "paper": "https://arxiv.org/abs/2305.06456",
        "code": "https://github.com/ZhengyiLuo/PHC",
        "group": "goal-conditioned",
        "note": "长期稳定 avatar 控制；身体行为连续性是 BFM 前置积累。",
        "wiki": ["wiki/entities/zhengyi-luo.md"],
    },
    {
        "id": 23,
        "slug": "teamplay_arxiv_2105_12196",
        "title": "TeamPlay: From Motor Control to Team Play in Simulated Humanoid Football",
        "year": 2021,
        "venue": "Science Robotics",
        "paper": "https://arxiv.org/abs/2105.12196",
        "code": "",
        "group": "goal-conditioned",
        "note": "底层 motor control 成熟后上层协作/战术才有空间。",
        "wiki": ["wiki/overview/humanoid-rl-motion-control-body-system-stack.md"],
    },
    {
        "id": 24,
        "slug": "mtm_arxiv_2305_02968",
        "title": "MTM: Masked Trajectory Models for Prediction, Representation, and Control",
        "year": 2023,
        "venue": "ICML",
        "paper": "https://arxiv.org/abs/2305.02968",
        "code": "https://github.com/facebookresearch/mtm",
        "group": "goal-conditioned",
        "note": "轨迹作为预训练对象；行为基座不必只靠传统 RL 交互。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 25,
        "slug": "ase_arxiv_2205_01906",
        "title": "ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters",
        "year": 2022,
        "venue": "TOG",
        "paper": "https://arxiv.org/abs/2205.01906",
        "code": "https://github.com/nv-tlabs/ASE",
        "group": "goal-conditioned",
        "note": "可复用技能嵌入；与 AMP 分布约束不同，强调技能空间。",
        "wiki": ["wiki/overview/humanoid-amp-motion-prior-survey.md"],
    },
    {
        "id": 26,
        "slug": "aps_icml_2021",
        "title": "Active Pretraining with Successor Features",
        "year": 2021,
        "venue": "ICML",
        "paper": "https://arxiv.org/abs/2106.14910",
        "code": "https://github.com/rll-research/url_benchmark",
        "group": "intrinsic-reward",
        "note": "任务到来前的主动预训练；低层身体需先积累可迁移经验。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 27,
        "slug": "proto_rl_icml_2021",
        "title": "Reinforcement Learning with Prototypical Representations",
        "year": 2021,
        "venue": "ICML",
        "paper": "https://arxiv.org/abs/2102.11271",
        "code": "https://github.com/denisyarats/proto",
        "group": "intrinsic-reward",
        "note": "原型表示组织行为空间，便于上层组合与选择技能。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 28,
        "slug": "re3_icml_2020",
        "title": "State Entropy Maximization with Random Encoders for Efficient Exploration",
        "year": 2020,
        "venue": "ICML",
        "paper": "https://arxiv.org/abs/2102.09430",
        "code": "https://github.com/younggyoseo/RE3",
        "group": "intrinsic-reward",
        "note": "状态熵最大化打开探索；人形需覆盖更多姿态以防 OOD 失败。",
        "wiki": ["wiki/methods/reinforcement-learning.md"],
    },
    {
        "id": 29,
        "slug": "rnd_iclr_2019",
        "title": "Exploration by Random Network Distillation",
        "year": 2019,
        "venue": "ICLR",
        "paper": "https://arxiv.org/abs/1810.12894",
        "code": "https://github.com/openai/random-network-distillation",
        "group": "intrinsic-reward",
        "note": "内在奖励探索经典；身体预训练常需自发现新状态。",
        "wiki": ["wiki/methods/reinforcement-learning.md"],
    },
    {
        "id": 30,
        "slug": "diayn_iclr_2018",
        "title": "Diversity is All You Need: Learning Skills without a Reward Function",
        "year": 2018,
        "venue": "ICLR",
        "paper": "https://arxiv.org/abs/1802.06070",
        "code": "https://github.com/alirezakazemipour/DIAYN-PyTorch",
        "group": "intrinsic-reward",
        "note": "无外部奖励学多样技能；「先学技能再服务任务」的早期思路。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 31,
        "slug": "task_tokens_arxiv_2503_22886",
        "title": "Task Tokens: A Flexible Approach to Adapting Behavior Foundation Models",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2503.22886",
        "code": "",
        "group": "adaptation",
        "note": "轻量 task token 适配，把重训身体转为换任务条件。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 32,
        "slug": "unseen_dynamics_arxiv_2505_13150",
        "title": "Zero-Shot Adaptation of Behavioral Foundation Models to Unseen Dynamics",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2505.13150",
        "code": "",
        "group": "adaptation",
        "note": "负载/地面/硬件参数变化下的零样本动力学适配。",
        "wiki": [
            "wiki/concepts/sim2real.md",
            "wiki/entities/paper-any2any-cross-embodiment-wbt.md",
        ],
    },
    {
        "id": 33,
        "slug": "fast_adaptation_bfm_corl_2025",
        "title": "Fast Adaptation With Behavioral Foundation Models",
        "year": 2025,
        "venue": "CoRL",
        "paper": "https://arxiv.org/abs/2504.07896",
        "code": "",
        "group": "adaptation",
        "note": "适配速度决定 BFM 工程价值：新任务样本与开发成本。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 34,
        "slug": "sentinel_arxiv_2511_19236",
        "title": "SENTINEL: A Fully End-to-End Language-Action Model for Humanoid Whole Body Control",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2511.19236",
        "code": "",
        "group": "hierarchical",
        "note": "语言–全身动作端到端；中间须有处理平衡/接触的身体通道。",
        "wiki": [
            "wiki/methods/vla.md",
            "wiki/overview/humanoid-rl-motion-control-body-system-stack.md",
        ],
    },
    {
        "id": 35,
        "slug": "beyondmimic_arxiv_2508_08241",
        "title": "BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2508.08241",
        "code": "https://github.com/HybridRobotics/whole_body_tracking",
        "group": "hierarchical",
        "note": "guided diffusion 进入全身控制；生成方案仍需低层执行器兜底。",
        "wiki": ["wiki/methods/beyondmimic.md"],
    },
    {
        "id": 36,
        "slug": "leverb_arxiv_2506_13751",
        "title": "LeVerb: Humanoid Whole-Body Control with Latent Vision-Language Instruction",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2506.13751",
        "code": "",
        "group": "hierarchical",
        "note": "视觉-语言指令压成适合全身的 latent；语义到身体接口。",
        "wiki": ["wiki/methods/vla.md", "wiki/entities/paper-daji-anticipatory-joint-intent.md"],
    },
    {
        "id": 37,
        "slug": "langwbc_arxiv_2504_21738",
        "title": "LangWBC: Language-Directed Humanoid Whole-Body Control via End-to-end Learning",
        "year": 2025,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2504.21738",
        "code": "",
        "group": "hierarchical",
        "note": "语言直接进入端到端 WBC；难在语义进入身体后不打散稳定性。",
        "wiki": ["wiki/methods/vla.md"],
    },
    {
        "id": 38,
        "slug": "tokenhsi_arxiv_2503_19901",
        "title": "Tokenhsi: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization",
        "year": 2025,
        "venue": "CVPR",
        "paper": "https://arxiv.org/abs/2503.19901",
        "code": "https://github.com/liangpan99/TokenHSI",
        "group": "hierarchical",
        "note": "人-场景交互 task token 化；坐下/跨越等是结构化事件。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 39,
        "slug": "closd_arxiv_2410_03441",
        "title": "CloSD: Closing the Loop between Simulation and Diffusion for Multi-task Character Control",
        "year": 2024,
        "venue": "ICLR",
        "paper": "https://arxiv.org/abs/2410.03441",
        "code": "https://github.com/GuyTevet/CLoSD",
        "group": "hierarchical",
        "note": "扩散生成与仿真闭环校正；生成动作须经物理检验。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 40,
        "slug": "uniphys_arxiv_2504_12540",
        "title": "UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-based Character Control",
        "year": 2024,
        "venue": "arXiv",
        "paper": "https://arxiv.org/abs/2504.12540",
        "code": "https://wuyan01.github.io/uniphys-project/",
        "group": "hierarchical",
        "note": "planner 与 controller 在扩散框架内协同；BFM 成熟后的分层边界问题。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "id": 41,
        "slug": "unihsi_arxiv_2309_07918",
        "title": "Unified Human-Scene Interaction via Prompted Chain-of-Contacts",
        "year": 2023,
        "venue": "ICLR",
        "paper": "https://arxiv.org/abs/2309.07918",
        "code": "https://github.com/OpenRobotLab/UniHSI",
        "group": "hierarchical",
        "note": "contact chain 组织交互；任务难在接触顺序而非单姿态。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
]

DATASETS: list[dict] = [
    {
        "slug": "dataset_humanoid_x_arxiv_2501_05098",
        "name": "Humanoid-X",
        "year": 2025,
        "venue": "Humanoids",
        "clips": "163800",
        "hours": "240.0",
        "paper": "https://arxiv.org/abs/2501.05098",
        "code": "https://github.com/sihengz02/UH-1",
        "note": "大规模人形动作数据入口，贴近 BFM goal-conditioned scaling。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "slug": "dataset_phuma_arxiv_2510_26236",
        "name": "PHUMA",
        "year": 2025,
        "venue": "arXiv",
        "clips": "76000",
        "hours": "73.0",
        "paper": "https://arxiv.org/abs/2510.26236",
        "code": "https://github.com/DAVIAN-Robotics/PHUMA",
        "note": "人形动作规模化与结构化。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "slug": "dataset_motion_xpp_arxiv_2501_05098",
        "name": "Motion-X++",
        "year": 2025,
        "venue": "arXiv",
        "clips": "120462",
        "hours": "180.9",
        "paper": "https://arxiv.org/abs/2501.05098",
        "code": "https://github.com/IDEA-Research/Motion-X",
        "note": "扩展 Motion-X 覆盖面。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "slug": "dataset_motion_x_neurips_2023",
        "name": "Motion-X",
        "year": 2023,
        "venue": "NeurIPS",
        "clips": "81084",
        "hours": "144.2",
        "paper": "https://proceedings.neurips.cc/paper_files/paper/2023/file/4f8e27f6036c1d8b4a66b5b3a947dd7b-Paper-Datasets_and_Benchmarks.pdf",
        "code": "https://github.com/IDEA-Research/Motion-X",
        "note": "文本–动作–大规模人体运动连接。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "slug": "dataset_posescript_eccv_2022",
        "name": "PoseScript",
        "year": 2022,
        "venue": "ECCV",
        "clips": "-",
        "hours": "-",
        "paper": "https://arxiv.org/abs/2210.11795",
        "code": "https://github.com/naver/posescript",
        "note": "语言描述与姿态桥接。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "slug": "dataset_humanml3d_cvpr_2022",
        "name": "HumanML3D",
        "year": 2022,
        "venue": "CVPR",
        "clips": "14616",
        "hours": "28.6",
        "paper": "https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf",
        "code": "https://github.com/EricGuo5513/HumanML3D",
        "note": "文本到人体动作生成常用基准。",
        "wiki": ["wiki/entities/amass.md"],
    },
    {
        "slug": "dataset_babel_cvpr_2021",
        "name": "BABEL",
        "year": 2021,
        "venue": "CVPR",
        "clips": "13220",
        "hours": "43.5",
        "paper": "https://arxiv.org/abs/2106.09696",
        "code": "https://babel.is.tue.mpg.de/",
        "note": "动作语义标注，连接语言与动作。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
    {
        "slug": "dataset_lafan_tog_2020",
        "name": "LAFAN",
        "year": 2020,
        "venue": "TOG",
        "clips": "77",
        "hours": "4.6",
        "paper": "https://arxiv.org/abs/2102.04942",
        "code": "https://github.com/ubisoft/ubisoft-laforge-animation-dataset",
        "note": "motion imitation 常用动作源。",
        "wiki": ["wiki/methods/deepmimic.md"],
    },
    {
        "slug": "dataset_amass_iccv_2019",
        "name": "AMASS",
        "year": 2019,
        "venue": "ICCV",
        "clips": "11265",
        "hours": "40.0",
        "paper": "https://arxiv.org/abs/1904.03278",
        "code": "https://amass.is.tue.mpg.de/",
        "note": "动捕大集合；人形重定向与 BFM 训练常见起点。",
        "wiki": ["wiki/entities/amass.md"],
        "skip": True,
        "existing": "sources/sites/amass-dataset.md",
    },
    {
        "slug": "dataset_kit_ml_arxiv_2016",
        "name": "KIT-ML",
        "year": 2016,
        "venue": "arXiv",
        "clips": "3911",
        "hours": "11.2",
        "paper": "https://arxiv.org/abs/1607.03827",
        "code": "https://motion-annotation.humanoids.kit.edu/dataset/",
        "note": "较早的语言–动作配对数据。",
        "wiki": ["wiki/concepts/behavior-foundation-model.md"],
    },
]

GROUP_LABEL = {
    "forward-backward": "01 Forward-backward 表征",
    "goal-conditioned": "02 Goal-conditioned 学习",
    "intrinsic-reward": "03 Intrinsic reward 预训练",
    "adaptation": "04 Adaptation",
    "hierarchical": "05 Hierarchical control",
}


def paper_md(p: dict) -> str:
    code_line = f"- **代码/项目：** <{p['code']}>\n" if p.get("code") else "- **代码/项目：** N/A\n"
    wiki_lines = "\n".join(
        f"  - [{w.split('/')[-1].replace('.md', '')}](../../{w})" for w in p["wiki"]
    )
    return f"""# {p["title"]}

> 来源归档（ingest · awesome-bfm-papers 第 {p["id"]:02d}/41）

- **标题：** {p["title"]}
- **类型：** paper
- **BFM 分类：** {GROUP_LABEL[p["group"]]}（[awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)）
- **出处：** {p["year"]} · {p["venue"]}
- **论文链接：** <{p["paper"]}>
{code_line}- **索引来源：** [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) · [具身智能研究室 BFM 41 篇编译](../blogs/{WECHAT})（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）
- **入库日期：** {TODAY}
- **一句话说明：** {p["note"]}

## 核心摘录（策展，非全文）

- **在 BFM 技术地图中的位置：** {GROUP_LABEL[p["group"]]}，编号 **{p["id"]:02d}/41**。
- **公众号导读要点：** {p["note"]}
- **读者动作：** 方法细节以论文 PDF / 项目页为准；谱系对照见 [BFM 41 篇技术地图](../../wiki/overview/bfm-41-papers-technology-map.md) 与 [Behavior Foundation Model](../../wiki/concepts/behavior-foundation-model.md)。

## 对 wiki 的映射

{wiki_lines}

## 参考来源（原始）

- 论文：<{p["paper"]}>
- 策展列表：<https://github.com/friedrichyuan/awesome-bfm-papers>
- 微信公众号编译：[wechat_embodied_ai_lab_bfm_41_papers_survey.md](../blogs/{WECHAT})
"""


def dataset_md(d: dict) -> str:
    wiki_lines = "\n".join(
        f"  - [{w.split('/')[-1].replace('.md', '')}](../../{w})" for w in d["wiki"]
    )
    existing = ""
    if d.get("existing"):
        existing = f"\n- **本库已有归档：** [`{d['existing']}`](../sites/amass-dataset.md)（本文件补 awesome-bfm 数据集表索引位）\n"
    return f"""# {d["name"]}（BFM 行为数据 · awesome-bfm-papers 数据集表）

> 来源归档（ingest · 数据集）

- **名称：** {d["name"]}
- **类型：** dataset
- **BFM 数据索引：** [awesome-bfm-papers § Datasets](https://github.com/friedrichyuan/awesome-bfm-papers#datasets)
- **出处：** {d["year"]} · {d["venue"]}
- **规模：** {d["clips"]} clips · {d["hours"]} h（列表标注）
- **论文链接：** <{d["paper"]}>
- **代码/入口：** <{d["code"]}>
- **索引来源：** [具身智能研究室 BFM 41 篇编译](../blogs/{WECHAT})
- **入库日期：** {TODAY}
- **一句话说明：** {d["note"]}{existing}

## 核心摘录（策展）

- BFM 数据关键不在条数，而在能否加工成 **机器人可信、可执行、可迁移** 的训练材料。
- 与 [BFM 41 篇技术地图](../../wiki/overview/bfm-41-papers-technology-map.md) § 数据集 互参。

## 对 wiki 的映射

{wiki_lines}

## 参考来源（原始）

- 数据集论文/页：<{d["paper"]}>
- awesome-bfm-papers：<https://github.com/friedrichyuan/awesome-bfm-papers>
"""


def catalog_md(papers_written: list[dict], datasets_written: list[dict]) -> str:
    lines = [
        "# awesome-bfm-papers：41 篇论文 + 10 数据集 source 索引",
        "",
        "> 来源归档（catalog）",
        "",
        "- **维护列表：** <https://github.com/friedrichyuan/awesome-bfm-papers>",
        f"- **微信公众号导读：** [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../blogs/{WECHAT})",
        "- **wiki 技术地图：** [bfm-41-papers-technology-map.md](../../wiki/overview/bfm-41-papers-technology-map.md)",
        f"- **入库日期：** {TODAY}",
        "- **一句话说明：** 将 awesome-bfm-papers 所列 **41 篇 BFM 论文** 与 **10 个数据集** 分别落成独立 `sources/papers/bfm_awesome_*` 归档，便于 ingest 溯源与 lint 入链统计。",
        "",
        "## 论文（41）",
        "",
        "| # | 分组 | Source |",
        "|---|------|--------|",
    ]
    for p in papers_written:
        lines.append(
            f"| {p['id']:02d} | {GROUP_LABEL[p['group']]} | [bfm_awesome_{p['slug']}.md](bfm_awesome_{p['slug']}.md) |"
        )
    lines.extend(["", "## 数据集（10）", "", "| 数据集 | Source |", "|--------|--------|"])
    for d in datasets_written:
        lines.append(f"| {d['name']} | [bfm_awesome_{d['slug']}.md](bfm_awesome_{d['slug']}.md) |")
    lines.extend(
        [
            "",
            "## 对 wiki 的映射",
            "",
            "- [behavior-foundation-model.md](../../wiki/concepts/behavior-foundation-model.md)",
            "- [bfm-41-papers-technology-map.md](../../wiki/overview/bfm-41-papers-technology-map.md)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    written_papers: list[dict] = []
    written_datasets: list[dict] = []
    created = 0
    skipped = 0

    for p in PAPERS:
        out = PAPERS_DIR / f"bfm_awesome_{p['slug']}.md"
        if p.get("skip") and (ROOT / "sources/papers" / f"{p['slug']}.md").exists():
            # cross-ref stub for #13
            if p["id"] == 13:
                stub = f"""# {p["title"]}（索引位 · 已有深读归档）

> 本条目对应 awesome-bfm-papers **第 13/41** 篇；完整 ingest 见既有归档。

- **主归档：** [bfm_humanoid_arxiv_2509_13780.md](bfm_humanoid_arxiv_2509_13780.md)
- **wiki：** [paper-behavior-foundation-model-humanoid.md](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)
- **BFM 分类：** {GROUP_LABEL[p["group"]]}
- **论文：** <{p["paper"]}>
"""
                out.write_text(stub, encoding="utf-8")
                created += 1
            skipped += 1
            written_papers.append(p)
            continue
        if out.exists():
            skipped += 1
            written_papers.append(p)
            continue
        out.write_text(paper_md(p), encoding="utf-8")
        created += 1
        written_papers.append(p)

    for d in DATASETS:
        out = PAPERS_DIR / f"bfm_awesome_{d['slug']}.md"
        if d.get("skip") and d.get("existing"):
            if out.exists():
                skipped += 1
            else:
                out.write_text(dataset_md(d), encoding="utf-8")
                created += 1
            written_datasets.append(d)
            continue
        if out.exists():
            skipped += 1
            written_datasets.append(d)
            continue
        out.write_text(dataset_md(d), encoding="utf-8")
        created += 1
        written_datasets.append(d)

    catalog_path = PAPERS_DIR / CATALOG
    catalog_path.write_text(catalog_md(written_papers, written_datasets), encoding="utf-8")
    print(f"created={created} skipped={skipped} catalog={catalog_path.name}")


if __name__ == "__main__":
    main()
