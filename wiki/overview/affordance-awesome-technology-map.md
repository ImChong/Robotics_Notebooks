---
type: overview
tags: [overview, curated-index, awesome-affordance-learning, affordance-learning, technology-map]
status: complete
updated: 2026-10-23
summary: "Awesome-Affordance-Learning 技术地图：清单中 206 篇去重论文逐条对应站内一页，按感知 / 推理 / 动作 / 数据集 / 综述分组。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Awesome-Affordance-Learning 技术地图

> 把 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning) 清单的论文逐条拆成站内可点开的一页，按清单分组浏览。该清单是 [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md) 的配套资源。

## 一句话定义

外部 Awesome 清单的站内版本：每篇论文一个节点，已有深读页的条目直接链过去，其余为索引级清单页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 原清单每条只有标题与链接，站内逐条成页后可检索、可互链。
- 同一论文在清单多个分组出现（如数据集 / 感知）时只保留一个节点；与站内既有页重复的条目复用既有页，不新建。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游仓库 | <https://github.com/hq-King/Awesome-Affordance-Learning> |
| 去重后条目 | 206 |
| 目录 source | [affordance_awesome_catalog.md](../../sources/papers/affordance_awesome_catalog.md) |
| 配套综述 | [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md) |

## 分组索引

### Perception

| 出处 | 论文 |
|------|------|
| IJRR 2013 | [Learning Human Activities and Object Affordances from RGB-D Videos](../entities/paper-aff-learning-human-activities-and-object-affordances-from-rg.md) |
| ICRA 2015 | [Affordance Detection of Tool Parts from Geometric Features](../entities/paper-aff-affordance-detection-of-tool-parts-from-geometric-featur.md) |
| ECCV 2016 | [A Multi-scale CNN for Affordance Segmentation in RGB Images](../entities/paper-aff-a-multi-scale-cnn-for-affordance-segmentation-in-rgb-ima.md) |
| IROS 2016 | [Detecting Object Affordances with Convolutional Neural Networks](../entities/paper-aff-detecting-object-affordances-with-convolutional-neural-n.md) |
| CVPR 2017 | [Weakly Supervised Affordance Detection](../entities/paper-aff-weakly-supervised-affordance-detection.md) |
| IROS 2017 | [Object-Based Affordances Detection with Convolutional Neural Networks and Dense Conditiona](../entities/paper-aff-object-based-affordances-detection-with-convolutional-ne.md) |
| ICCVW 2017 | [Adaptive Binarization for Weakly Supervised Affordance Segmentation](../entities/paper-aff-1707-02850-adaptive-binarization-for-weakly-supervised-affordance-s.md) |
| ICCVW 2017 | [Learning to Segment Affordances](../entities/paper-aff-learning-to-segment-affordances.md) |
| Humanoids 2017 | [Affordance Detection for Task-Specific Grasping Using Deep Learning](../entities/paper-aff-affordance-detection-for-task-specific-grasping-using-de.md) |
| ICRA 2018 | [AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection](../entities/paper-aff-1709-07326-affordancenet-an-end-to-end-deep-learning-approach-for-o.md) |
| CVPR 2018 | [Demo2Vec: Reasoning Object Affordances from Online Videos](../entities/paper-aff-demo2vec-reasoning-object-affordances-from-online-videos.md) |
| ECCV 2018 | [In the Eye of Beholder: Joint Learning of Gaze and Actions in First Person Video](../entities/paper-aff-in-the-eye-of-beholder-joint-learning-of-gaze-and-action.md) |
| Autonomous Robots 2019 | [Towards Affordance Detection for Robot Manipulation Using Affordance for Parts and Parts f](../entities/paper-aff-towards-affordance-detection-for-robot-manipulation-usin.md) |
| RAS 2019 | [Context-Based Affordance Segmentation from 2D Images for Robot Actions](../entities/paper-aff-context-based-affordance-segmentation-from-2d-images-for.md) |
| RA-L with ICRA 2019 | [Learning Affordance Segmentation for Real-world Robotic Manipulation via Synthetic Images](../entities/paper-aff-learning-affordance-segmentation-for-real-world-robotic.md) |
| arXiv 2019 | [Recognizing Object Affordances to Support Scene Reasoning for Manipulation Tasks](../entities/paper-aff-1909-05770-recognizing-object-affordances-to-support-scene-reasonin.md) |
| ICCV 2019 | [Grounded Human-Object Interaction Hotspots From Video](../entities/paper-aff-grounded-human-object-interaction-hotspots-from-video.md) |
| CVPR 2020 | [Ego-Topo: Environment Affordances from Egocentric Video](../entities/paper-aff-2001-04583-ego-topo-environment-affordances-from-egocentric-video.md) |
| NeurIPS 2020 | [Learning Affordance Landscapes for Interaction Exploration in 3D Environments](../entities/paper-aff-learning-affordance-landscapes-for-interaction-explorati.md) |
| Neural Comput. Appl. 2020.09 | [Object Affordance Detection with Relationship-Aware Network](../entities/paper-aff-object-affordance-detection-with-relationship-aware-netw.md) |
| RA-L with ICRA2021 | [An Affordance Keypoint Detection Network for Robot Manipulation](../entities/paper-aff-an-affordance-keypoint-detection-network-for-robot-manip.md) |
| ACM ICDAR 2021 | [ST-HOI: A Spatial-Temporal Baseline for Human-Object Interaction Detection in Videos](../entities/paper-aff-st-hoi-a-spatial-temporal-baseline-for-human-object-inte.md) |
| CVPR 2021 | [3D AffordanceNet: A Benchmark for Visual Object Affordance Understanding](../entities/paper-aff-2103-16397-3d-affordancenet-a-benchmark-for-visual-object-affordanc.md) |
| CoRL 2021 | [O2O-Afford: Annotation-Free Large-Scale Object-Object Affordance Learning](../entities/paper-aff-2106-15087-o2o-afford-annotation-free-large-scale-object-object-aff.md) |
| Neurocomputing 2021 | [Visual Affordance Detection Using an Efficient Attention Convolutional Neural Network](../entities/paper-aff-visual-affordance-detection-using-an-efficient-attention.md) |
| IJCAI 2021 | [One-Shot Affordance Detection](../entities/paper-aff-2106-14747-one-shot-affordance-detection.md) |
| Humanoids 2022 | [Manipulation-Oriented Object Perception in Clutter through Affordance Coordinate Frames](../entities/paper-aff-2010-08202-manipulation-oriented-object-perception-in-clutter-throu.md) |
| ECCVW 2022 | [PartAfford: Part-level Affordance Discovery from 3D Objects](../entities/paper-aff-2202-13519-partafford-part-level-affordance-discovery-from-3d-objec.md) |
| CVPR 2022 | [Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos](../entities/paper-aff-2204-01696-joint-hand-motion-and-interaction-hotspots-prediction-fr.md) |
| CVPR 2022 | [Learning Affordance Grounding from Exocentric Images](../entities/paper-aff-2203-09905-learning-affordance-grounding-from-exocentric-images.md) |
| CoRL 2022 | [One-Shot Transfer of Affordance Regions? AffCorrs!](../entities/paper-aff-2209-07147-one-shot-transfer-of-affordance-regions-affcorrs.md) |
| Neural Comput. Appl. 2022 | [Object Affordance Detection with Boundary-Preserving Network for Robotic Manipulation Task](../entities/paper-aff-object-affordance-detection-with-boundary-preserving-net.md) |
| IJCV 2022 | [One-Shot Object Affordance Detection in the Wild](../entities/paper-aff-2108-03658-one-shot-object-affordance-detection-in-the-wild.md) |
| WACV 2023 | [Fine-Grained Affordance Annotation for Egocentric Hand-Object Interaction Videos](../entities/paper-aff-fine-grained-affordance-annotation-for-egocentric-hand-o.md) |
| arXiv 2023 | [STRAP: Structured Object Affordance Segmentation with Point Supervision](../entities/paper-aff-2304-08492-strap-structured-object-affordance-segmentation-with-poi.md) |
| CVPR 2023 | [Affordance Grounding From Demonstration Video To Target Image](../entities/paper-aff-2303-14644-affordance-grounding-from-demonstration-video-to-target.md) |
| CVPR 2023 | [Leverage Interactive Affinity for Affordance Learning](../entities/paper-aff-leverage-interactive-affinity-for-affordance-learning.md) |
| CVPR 2023 | [LOCATE: Localize and Transfer Object Parts for Weakly Supervised Affordance Grounding](../entities/paper-aff-2303-09665-locate-localize-and-transfer-object-parts-for-weakly-sup.md) |
| NeurIPS 2023 | [Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Obj](../entities/paper-aff-2309-07473-where2explore-few-shot-affordance-learning-for-unseen-no.md) |
| ICIP 2023 | [A Large Scale Multi-View RGBD Visual Affordance Learning Dataset](../entities/paper-aff-2203-14092-a-large-scale-multi-view-rgbd-visual-affordance-learning.md) |
| IROS 2023 | [Hierarchical Transformer for Visual Affordance Understanding using a Large-scale Dataset](../entities/paper-aff-hierarchical-transformer-for-visual-affordance-understan.md) |
| ICCV 2023 | [AffordPose: A Large-Scale Dataset of Hand-Object Interactions with Affordance-Driven Hand ](../entities/paper-aff-affordpose-a-large-scale-dataset-of-hand-object-interact.md) |
| IROS 2023 | [HANDAL: A Dataset of Real-World Manipulable Object Categories with Pose Annotations, Affor](../entities/paper-aff-2308-01477-handal-a-dataset-of-real-world-manipulable-object-catego.md) |
| ICCV 2023 | [Grounding 3D Object Affordance from 2D Interactions in Images](../entities/paper-aff-2303-10437-grounding-3d-object-affordance-from-2d-interactions-in-i.md) |
| ICCV 2023 | [Understanding 3D Object Interaction from a Single Image](../entities/paper-aff-understanding-3d-object-interaction-from-a-single-image.md) |
| ICCV 2023 | [Multi-label Affordance Mapping from Egocentric Vision](../entities/paper-aff-multi-label-affordance-mapping-from-egocentric-vision.md) |
| IROS 2023 | [Open-vocabulary Affordance Detection in 3D Point Clouds](../entities/paper-aff-2303-02401-open-vocabulary-affordance-detection-in-3d-point-clouds.md) |
| ICRA 2024 | [Open-Vocabulary Affordance Detection using Knowledge Distillation and Text-Point Correlati](../entities/paper-aff-2309-10932-open-vocabulary-affordance-detection-using-knowledge-dis.md) |
| CVPR 2024 | [LEMON: Learning 3D Human-Object Interaction Relation from 2D Images](../entities/paper-aff-2312-08963-lemon-learning-3d-human-object-interaction-relation-from.md) |
| AAAI 2024 | [Weakly Supervised Multimodal Affordance Grounding for Egocentric Images](../entities/paper-aff-weakly-supervised-multimodal-affordance-grounding-for-eg.md) |
| arXiv 2024 | [Text-driven Affordance Learning from Egocentric Vision](../entities/paper-aff-2404-02523-text-driven-affordance-learning-from-egocentric-vision.md) |
| ICRA 2024 | [Language-Conditioned Affordance-Pose Detection in 3D Point Clouds](../entities/paper-aff-language-conditioned-affordance-pose-detection-in-3d-poi.md) |
| CVPR 2024 | [One-Shot Open Affordance Learning with Foundation Models](../entities/paper-aff-2311-17776-one-shot-open-affordance-learning-with-foundation-models.md) |
| CVPR 2024 | [LASO: Language-guided Affordance Segmentation on 3D Object](../entities/paper-aff-laso-language-guided-affordance-segmentation-on-3d-objec.md) |
| CVPRW 2024 | [LGAfford-Net: A Local Geometry Aware Affordance Detection Network for 3D Point Clouds](../entities/paper-aff-lgafford-net-a-local-geometry-aware-affordance-detection.md) |
| CVPR 2024 | [SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D Scenes](../entities/paper-aff-scenefun3d-fine-grained-functionality-and-affordance-und.md) |
| arXiv 2024 | [Grounding 3D Scene Affordance From Egocentric Interactions](../entities/paper-aff-2409-19650-grounding-3d-scene-affordance-from-egocentric-interactio.md) |
| ECCV 2024 | [INTRA: Interaction Relationship-aware Weakly Supervised Affordance Grounding](../entities/paper-aff-2409-06210-intra-interaction-relationship-aware-weakly-supervised-a.md) |
| ECCV 2024 | [Beyond the Contact: Discovering Comprehensive Affordance for 3D Objects from Pre-trained 2](../entities/paper-aff-beyond-the-contact-discovering-comprehensive-affordance.md) |
| IROS 2024 | [3D Affordance Keypoint Detection for Robotic Manipulation](../entities/paper-aff-3d-affordance-keypoint-detection-for-robotic-manipulatio.md) |
| ICCV 2025 | [GLANCE: Intermediate Connectors and Geometric Priors for Language-Guided Affordance Segmen](../entities/paper-aff-glance-intermediate-connectors-and-geometric-priors-for.md) |
| AAAI 2025 | [Learning 2D Invariant Affordance Knowledge for 3D Affordance Grounding](../entities/paper-aff-2408-13024-learning-2d-invariant-affordance-knowledge-for-3d-afford.md) |
| CVPR 2025 | [GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency](../entities/paper-aff-2412-09511-geal-generalizable-3d-affordance-learning-with-cross-mod.md) |
| AAAI 2025 | [MaskPrompt: Open-Vocabulary Affordance Segmentation with Object Shape Mask Prompts](../entities/paper-aff-maskprompt-open-vocabulary-affordance-segmentation-with.md) |
| arXiv 2025 | [AffordanceSAM: Segment Anything Once More in Affordance Grounding](../entities/paper-aff-2504-15650-affordancesam-segment-anything-once-more-in-affordance-g.md) |
| CVPR 2025 | [IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](../entities/paper-aff-2504-06827-iaao-interactive-affordance-learning-for-articulated-obj.md) |
| arXiv 2025 | [Interpretable Affordance Detection on 3D Point Clouds with Probabilistic Prototypes](../entities/paper-aff-2504-18355-interpretable-affordance-detection-on-3d-point-clouds-wi.md) |
| arXiv 2025 | [Weakly-Supervised Affordance Grounding Guided by Part-Level Semantic Priors](../entities/paper-aff-2505-24103-weakly-supervised-affordance-grounding-guided-by-part-le.md) |
| ICRA 2025 | [UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation](../entities/paper-aff-2506-09284-uad-unsupervised-affordance-distillation-for-generalizat.md) |
| CVPR 2025 | [Reasoning Mamba: Hypergraph-Guided Region Relation Calculating for Weakly Supervised Affor](../entities/paper-aff-reasoning-mamba-hypergraph-guided-region-relation-calcul.md) |
| IROS 2025 | [One-Shot Affordance Grounding of Deformable Objects in Egocentric Organizing Scenes](../entities/paper-aff-2503-01092-one-shot-affordance-grounding-of-deformable-objects-in-e.md) |
| IROS 2025 | [Resource-Efficient Affordance Grounding with Complementary Depth and Semantic Prompts](../entities/paper-aff-2503-02600-resource-efficient-affordance-grounding-with-complementa.md) |
| ICCV 2025 | [Closed-Loop Transfer for Weakly-supervised Affordance Grounding](../entities/paper-aff-2510-17384-closed-loop-transfer-for-weakly-supervised-affordance-gr.md) |
| ICCV 2025 | [Selective Contrastive Learning for Weakly Supervised Affordance Grounding](../entities/paper-aff-selective-contrastive-learning-for-weakly-supervised-aff.md) |
| ICCV 2025 | [OVA-Fields: Weakly Supervised Open-Vocabulary Affordance Fields for Robot Operational Part](../entities/paper-aff-ova-fields-weakly-supervised-open-vocabulary-affordance.md) |
| CVPR 2025 | [Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](../entities/paper-aff-2503-19199-open-vocabulary-functional-3d-scene-graphs-for-real-worl.md) |
| ACM MM 2025 | [RoboAfford: A Dataset and Benchmark for Enhancing Object and Spatial Affordance Learning i](../entities/paper-aff-roboafford-a-dataset-and-benchmark-for-enhancing-object.md) |
| ECCV 2026 | [Affogato: Learning Open-Vocabulary Affordance Grounding with Automated Data Generation at ](../entities/paper-aff-2506-12009-affogato-learning-open-vocabulary-affordance-grounding-w.md) |
| CVPR 2026 | [Affostruction: 3D Affordance Grounding with Generative Reconstruction](../entities/paper-aff-2601-09211-affostruction-3d-affordance-grounding-with-generative-re.md) |
| CVPR 2026 | [HAMMER: Harnessing MLLMs via Cross-Modal Integration for Intention-Driven 3D Affordance Gr](../entities/paper-aff-2603-02329-hammer-harnessing-mllms-via-cross-modal-integration-for.md) |
| arXiv 2026 | [PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](../entities/paper-aff-2603-09760-panoaffordancenet-towards-holistic-affordance-grounding.md) |
| ECCV 2026 | [Panoramic Affordance Prediction](../entities/paper-aff-2603-15558-panoramic-affordance-prediction.md) |
| CVPR 2026 | [AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers](../entities/paper-aff-2603-27970-affordmatcher-affordance-learning-in-3d-scenes-from-visu.md) |
| ECCV 2026 | [Diffusion Models are Open-World Affordance Learners: Leveraging Generative Priors for 3D A](../entities/paper-aff-2508-01651-diffusion-models-are-open-world-affordance-learners-leve.md) |
| CVPR 2026 | [FunREC: Reconstructing Functional 3D Scenes from Egocentric Interaction Videos](../entities/paper-sa-2604-05621-funrec-reconstructing-functional-3d-scenes-from.md) |
| arXiv 2026 | [AFUN: Towards an Affordance Foundation Model for Functionality Understanding](../entities/paper-aff-2606-02551-afun-towards-an-affordance-foundation-model-for-function.md) |

### Reasoning

| 出处 | 论文 |
|------|------|
| ECCV 2014 | [Reasoning about Object Affordances in a Knowledge Base Representation](../entities/paper-aff-reasoning-about-object-affordances-in-a-knowledge-base-r.md) |
| TCDS 2016 | [Bootstrapping Relational Affordances of Object Pairs Using Transfer](../entities/paper-aff-bootstrapping-relational-affordances-of-object-pairs-usi.md) |
| CVPR 2018 | [Learning to Act Properly: Predicting and Explaining Affordances From Images](../entities/paper-aff-learning-to-act-properly-predicting-and-explaining-affor.md) |
| RA-L/IROS 2019 | [Learning Grasp Affordance Reasoning through Semantic Relations](../entities/paper-aff-1906-09836-learning-grasp-affordance-reasoning-through-semantic-rel.md) |
| ICRA 2020 | [Learning Object Placements For Relational Instructions by Hallucinating Scene Representati](../entities/paper-aff-2001-08481-learning-object-placements-for-relational-instructions-b.md) |
| arXiv 2021 | [Relationship Oriented Affordance Learning through Manipulation Graph Construction](../entities/paper-aff-2110-14137-relationship-oriented-affordance-learning-through-manipu.md) |
| ICDH 2022 | [A Visual Affordance Reasoning Network Based on Graph Attention](../entities/paper-aff-a-visual-affordance-reasoning-network-based-on-graph-att.md) |
| CoRL 2023 | [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](../entities/paper-saycan.md) |
| NeurIPS 2023 | [Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents](../entities/paper-aff-2303-00855-grounded-decoding-guiding-text-generation-with-grounded.md) |
| CVPRW 2024 | [AffordanceLLM: Grounding Affordance from Vision Language Models](../entities/paper-aff-2401-06341-affordancellm-grounding-affordance-from-vision-language.md) |
| RA-L 2024 | [NaturalVLM: Leveraging Fine-grained Natural Language for Affordance-Guided Visual Manipula](../entities/paper-aff-2403-08355-naturalvlm-leveraging-fine-grained-natural-language-for.md) |
| ICRAW 2024 | [OVAL-Prompt: Open-Vocabulary Affordance Localization for Robot Manipulation through LLM Af](../entities/paper-aff-2404-11000-oval-prompt-open-vocabulary-affordance-localization-for.md) |
| ICTAI 2024 | [WorldAfford: Affordance Grounding based on Natural Language Instructions](../entities/paper-aff-2405-12461-worldafford-affordance-grounding-based-on-natural-langua.md) |
| CoRL 2024 | [RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics](../entities/paper-robopoint.md) |
| IROS 2024 | [ManipVQA: Injecting Robotic Affordance and Physically Grounded Information into Multi-Moda](../entities/paper-aff-2403-11289-manipvqa-injecting-robotic-affordance-and-physically-gro.md) |
| IROS 2025 | [PAVLM: Advancing Point Cloud based Affordance Understanding Via Vision-Language Model](../entities/paper-aff-2410-11564-pavlm-advancing-point-cloud-based-affordance-understandi.md) |
| CVPR 2025 | [GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance](../entities/paper-aff-2411-19626-great-geometry-intention-collaborative-inference-for-ope.md) |
| CVPR 2025 | [SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](../entities/paper-aff-2412-01550-seqafford-sequential-3d-affordance-reasoning-via-multimo.md) |
| ICLR 2025 | [3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detectio](../entities/paper-aff-2502-20041-3d-affordancellm-harnessing-large-language-models-for-op.md) |
| arXiv 2025 | [Afford-X: Generalizable and Slim Affordance Reasoning for Task-oriented Manipulation](../entities/paper-aff-2503-03556-afford-x-generalizable-and-slim-affordance-reasoning-for.md) |
| CVPR 2025 | [Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interac](../entities/paper-aff-2504-04744-grounding-3d-object-affordance-with-language-instruction.md) |
| ACM MM 2025 | [3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians](../entities/paper-aff-2504-11218-3daffordsplat-efficient-affordance-reasoning-with-3d-gau.md) |
| ICRA 2025 | [UniAff: A Unified Representation of Affordances for Tool Usage and Articulation with Visio](../entities/paper-aff-2409-20551-uniaff-a-unified-representation-of-affordances-for-tool.md) |
| ACL 2025 | [InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](../entities/paper-aff-2505-18291-instructpart-task-oriented-part-segmentation-with-instru.md) |
| CVPR 2025 | [Functionality Understanding and Segmentation in 3D Scenes](../entities/paper-aff-2411-16310-functionality-understanding-and-segmentation-in-3d-scene.md) |
| ICCV 2025 | [RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Gras](../entities/paper-aff-2507-23734-ragnet-large-scale-reasoning-based-affordance-segmentati.md) |
| arXiv 2025 | [SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting](../entities/paper-aff-2507-23772-seqaffordsplat-scene-level-sequential-affordance-reasoni.md) |
| arXiv 2025 | [Agentic Scene Policies: Unifying Space, Semantics, and Affordances for Robot Action](../entities/paper-aff-2509-19571-agentic-scene-policies-unifying-space-semantics-and-affo.md) |
| ACM MM 2025 | [Open-Vocabulary 3D Affordance Understanding via Functional Text Enhancement and Multilevel](../entities/paper-aff-open-vocabulary-3d-affordance-understanding-via-function.md) |
| IROS 2025 | [AffordGrasp: In-Context Affordance Reasoning for Open-Vocabulary Task-Oriented Grasping in](../entities/paper-aff-affordgrasp-in-context-affordance-reasoning-for-open-voc.md) |
| NeurIPS 2025 | [AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](../entities/paper-aff-2511-10017-affordbot-3d-fine-grained-embodied-reasoning-via-multimo.md) |
| IROSW 2025 | [RoboAfford++: A Generative AI-Enhanced Dataset for Multimodal Affordance Learning in Robot](../entities/paper-aff-2511-12436-roboafford-a-generative-ai-enhanced-dataset-for-multimod.md) |
| NeurIPS 2025 | [ViSPLA: Visual Iterative Self-Prompting for Language-Guided 3D Affordance Learning](../entities/paper-aff-vispla-visual-iterative-self-prompting-for-language-guid.md) |
| ICASSP 2026 | [Affordance Benchmark for MLLMs](../entities/paper-aff-2506-00893-affordance-benchmark-for-mllms.md) |
| AAAI 2026 | [Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal](../entities/paper-aff-2508-06206-affordance-r1-reinforcement-learning-for-generalizable-a.md) |
| AAAI 2026 | [Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement](../entities/paper-aff-2511-11702-task-aware-3d-affordance-segmentation-via-2d-guidance-an.md) |
| ECCV 2026 | [A4-Agent: An Agentic Framework for Zero-Shot Affordance Reasoning](../entities/paper-aff-2512-14442-a4-agent-an-agentic-framework-for-zero-shot-affordance-r.md) |
| arXiv 2026 | [TRACER: Texture-Robust Affordance Chain-of-Thought for Deformable-Object Refinement](../entities/paper-aff-2601-20208-tracer-texture-robust-affordance-chain-of-thought-for-de.md) |
| arXiv 2026 | [AffordanceGrasp-R1: Leveraging Reasoning-Based Affordance Segmentation with Reinforcement ](../entities/paper-aff-2602-03547-affordancegrasp-r1-leveraging-reasoning-based-affordance.md) |
| arXiv 2026 | [VideoAfford: Grounding 3D Affordance from Human-Object-Interaction Videos via Multimodal L](../entities/paper-aff-2602-09638-videoafford-grounding-3d-affordance-from-human-object-in.md) |
| arXiv 2026 | [SceneTeract: Agentic Functional Affordances and VLM Grounding in 3D Scenes](../entities/paper-aff-2603-29798-sceneteract-agentic-functional-affordances-and-vlm-groun.md) |
| arXiv 2026 | [Part-Aware Open-Vocabulary 3D Affordance Grounding via Prototypical Semantic and Geometric](../entities/paper-aff-2603-17647-part-aware-open-vocabulary-3d-affordance-grounding-via-p.md) |
| arXiv 2026 | [A3R: Agentic Affordance Reasoning via Cross-Dimensional Evidence in 3D Gaussian Scenes](../entities/paper-aff-2604-01882-a3r-agentic-affordance-reasoning-via-cross-dimensional-e.md) |
| ECCV 2026 | [Token-Based Affordance Grounding with Large Vision-Language Models](../entities/paper-aff-2607-03595-token-based-affordance-grounding-with-large-vision-langu.md) |
| arXiv 2026 | [CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects](../entities/paper-aff-2604-02060-compassad-intent-driven-3d-affordance-grounding-in-funct.md) |
| arXiv 2026 | [Affordance Agent Harness: Verification-Gated Skill Orchestration](../entities/paper-aff-2605-00663-affordance-agent-harness-verification-gated-skill-orches.md) |
| arXiv 2026 | [Grounding by Remembering: Cross-Scene and In-Scene Memory for 3D Functional Affordances](../entities/paper-aff-2605-11616-grounding-by-remembering-cross-scene-and-in-scene-memory.md) |
| arXiv 2026 | [UNIAFFORD: Token-Routed Multitask Learning for Generalizable 2D-3D Affordance Perception](../entities/paper-aff-2609-37264-uniafford-token-routed-multitask-learning-for-generaliza.md) |

### Action

| 出处 | 论文 |
|------|------|
| IEEE TCDS 2016 | [Training Agents With Interactive Reinforcement Learning and Contextual Affordances](../entities/paper-aff-training-agents-with-interactive-reinforcement-learning.md) |
| ICRA 2020 | [Learning Affordance Space in Physical World for Vision-based Robotic Object Manipulation](../entities/paper-aff-learning-affordance-space-in-physical-world-for-vision-b.md) |
| ICRA 2021 | [Learning Dexterous Grasping with Object-Centric Visual Affordances](../entities/paper-aff-2009-01439-learning-dexterous-grasping-with-object-centric-visual-a.md) |
| ICRA 2021 | [What Can I Do Here? Learning New Skills by Imagining Visual Affordances](../entities/paper-aff-2106-00671-what-can-i-do-here-learning-new-skills-by-imagining-visu.md) |
| ICRA 2021 | [Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes](../entities/painode-214-contactgraspnet.md) |
| ICCV 2021 | [Where2Act: From Pixels to Actions for Articulated 3D Objects](../entities/paper-aff-2101-02692-where2act-from-pixels-to-actions-for-articulated-3d-obje.md) |
| ECCV 2022 | [AdaAfford: Learning to Adapt Manipulation Affordance for 3D Articulated Objects via Few-sh](../entities/paper-aff-2112-00246-adaafford-learning-to-adapt-manipulation-affordance-for.md) |
| ICRA 2022 | [Affordance Learning from Play for Sample-Efficient Policy Learning](../entities/paper-aff-2203-00352-affordance-learning-from-play-for-sample-efficient-polic.md) |
| IROS 2022 | [Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visual Affordance](../entities/paper-aff-2210-08537-learning-6-dof-task-oriented-grasp-detection-via-implici.md) |
| ICLR 2022 | [VAT-Mart: Learning Visual Action Trajectory Proposals for Manipulating 3D Articulated Obje](../entities/paper-aff-2106-14440-vat-mart-learning-visual-action-trajectory-proposals-for.md) |
| ICRA 2023 | [Grounding Language with Visual Affordances over Unstructured Data](../entities/paper-aff-2210-01911-grounding-language-with-visual-affordances-over-unstruct.md) |
| ICLR 2023 | [DualAfford: Learning Collaborative Visual Affordance for Dual-gripper Object Manipulation](../entities/paper-aff-2207-01971-dualafford-learning-collaborative-visual-affordance-for.md) |
| ICRA 2023 | [RLAfford: End-to-End Affordance Learning for Robotic Manipulation](../entities/paper-aff-2209-12941-rlafford-end-to-end-affordance-learning-for-robotic-mani.md) |
| ICRA 2023 | [Visual Affordance Prediction for Guiding Robot Exploration](../entities/paper-aff-2305-17783-visual-affordance-prediction-for-guiding-robot-explorati.md) |
| CoRL 2023 | [Affordance-Driven Next-Best-View Planning for Robotic Grasping](../entities/paper-aff-2309-09556-affordance-driven-next-best-view-planning-for-robotic-gr.md) |
| NeurIPS 2023 | [Learning Environment-aware Affordance for 3D Articulated Object Manipulation under Occlusi](../entities/paper-aff-2309-07510-learning-environment-aware-affordance-for-3d-articulated.md) |
| CoRL 2023 | [Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping](../entities/paper-aff-2309-07970-language-embedded-radiance-fields-for-zero-shot-task-ori.md) |
| CoRL 2023 | [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models](../entities/paper-voxposer.md) |
| CVPR 2023 | [Affordances from Human Videos as a Versatile Representation for Robotics](../entities/paper-aff-2304-08488-affordances-from-human-videos-as-a-versatile-representat.md) |
| ICCV 2023 | [Learning Foresightful Dense Visual Affordance for Deformable Object Manipulation](../entities/paper-aff-2303-11057-learning-foresightful-dense-visual-affordance-for-deform.md) |
| IEEE T-RO 2024 | [PRIMP: Probabilistically-Informed Motion Primitives for Efficient Affordance Learning from](../entities/paper-aff-primp-probabilistically-informed-motion-primitives-for-e.md) |
| arXiv 2024 | [Open-Vocabulary Part-Based Grasping](../entities/paper-aff-2406-05951-open-vocabulary-part-based-grasping.md) |
| ICRA 2024 | [Information-driven Affordance Discovery for Efficient Robotic Manipulation](../entities/paper-aff-information-driven-affordance-discovery-for-efficient-ro.md) |
| arXiv 2024 | [Affordance-based Robot Manipulation with Flow Matching](../entities/paper-aff-2409-01083-affordance-based-robot-manipulation-with-flow-matching.md) |
| CoRL 2024 | [ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulati](../entities/paper-aff-2409-01652-rekep-spatio-temporal-reasoning-of-relational-keypoint-c.md) |
| ECCV 2024 | [Robo-ABC: Affordance Generalization Beyond Categories via Semantic Correspondence for Robo](../entities/paper-aff-2401-07487-robo-abc-affordance-generalization-beyond-categories-via.md) |
| IROS 2024 | [PreAfford: Universal Affordance-Based Pre-Grasping for Diverse Objects and Environments](../entities/paper-aff-2404-03634-preafford-universal-affordance-based-pre-grasping-for-di.md) |
| CoRL 2024 | [RAM: Retrieval-Based Affordance Transfer for Generalizable Zero-Shot Robotic Manipulation](../entities/paper-aff-ram-retrieval-based-affordance-transfer-for-generalizabl.md) |
| ICCV 2025 | [Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](../entities/paper-aff-learning-precise-affordances-from-egocentric-videos-for.md) |
| ICRA 2025 | [RT-Affordance: Affordances are Versatile Intermediate Representations for Robot Manipulati](../entities/paper-aff-2411-02704-rt-affordance-affordances-are-versatile-intermediate-rep.md) |
| IROS 2025 | [ManipGPT: Is Affordance Segmentation by Large Vision Models Enough for Articulated Object ](../entities/paper-aff-2412-10050-manipgpt-is-affordance-segmentation-by-large-vision-mode.md) |
| ICCV 2025 | [AffordDexGrasp: Open-set Language-guided Dexterous Grasp with Generalizable-Instructive Af](../entities/paper-aff-2503-07360-afforddexgrasp-open-set-language-guided-dexterous-grasp.md) |
| CVPR 2025 | [GarmentPile: Point-Level Visual Affordance Guided Retrieval and Adaptation for Cluttered G](../entities/paper-aff-2503-09243-garmentpile-point-level-visual-affordance-guided-retriev.md) |
| ICCV 2025 | [A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](../entities/paper-rcl-ref-4482db2c9a502c2e3bae-a0-an-affordance-aware-hierarchical-model-for-ge.md) |
| IEEE CBS 2025 | [DORA: Object Affordance-Guided Reinforcement Learning for Dexterous Robotic Manipulation](../entities/paper-aff-2505-14819-dora-object-affordance-guided-reinforcement-learning-for.md) |
| CoRL 2025 | [GLOVER++: Unleashing the Potential of Affordance Learning from Human Behaviors for Robotic](../entities/paper-aff-2505-11865-glover-unleashing-the-potential-of-affordance-learning-f.md) |
| CVPR 2025 | [AffordDP: Generalizable Diffusion Policy with Transferable Affordance](../entities/paper-aff-2412-03142-afforddp-generalizable-diffusion-policy-with-transferabl.md) |
| ICML 2025 | [BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly](../entities/paper-aff-2506-06221-biassemble-learning-collaborative-affordance-for-bimanua.md) |
| CoRL 2025 | [O3Afford: One-Shot 3D Object-to-Object Affordance Grounding for Generalizable Robotic Mani](../entities/paper-aff-2509-06233-o3afford-one-shot-3d-object-to-object-affordance-groundi.md) |
| IEEE/ASME Transactions on Mechatronics 2025 | [DART: A Novel Task-Driven Diffusion-Based Policy with Affordance Learning for Generalizabl](../entities/paper-aff-2509-14939-dart-a-novel-task-driven-diffusion-based-policy-with-aff.md) |
| ICCV 2025 | [CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](../entities/paper-aff-2412-20451-coa-vla-improving-vision-language-action-models-via-visu.md) |
| ICCV 2025 | [2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](../entities/paper-aff-2handedafforder-learning-precise-actionable-bimanual-aff.md) |
| arXiv 2025 | [OVAL-Grasp: Open-Vocabulary Affordance Localization for Task Oriented Grasping](../entities/paper-aff-2511-20841-oval-grasp-open-vocabulary-affordance-localization-for-t.md) |
| RA-L 2025 | [ScaleADFG: Affordance-based Dexterous Functional Grasping via Scalable Dataset](../entities/paper-aff-2511-09602-scaleadfg-affordance-based-dexterous-functional-grasping.md) |
| AAAI 2026 | [Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors](../entities/paper-aff-2508-08896-towards-affordance-aware-robotic-dexterous-grasping-with.md) |
| CVPR 2026 | [Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulatio](../entities/paper-aff-2512-07472-affordance-field-intervention-enabling-vlas-to-escape-me.md) |
| AAAI 2026 | [A3D: Adaptive Affordance Assembly with Dual-Arm Manipulation](../entities/paper-aff-2601-11076-a3d-adaptive-affordance-assembly-with-dual-arm-manipulat.md) |
| IROS 2026 | [FSAG: Enhancing Human-to-Dexterous-Hand Finger-Specific Affordance Grounding via Diffusion](../entities/paper-aff-2601-08246-fsag-enhancing-human-to-dexterous-hand-finger-specific-a.md) |
| arXiv 2026 | [Generate, Transfer, Adapt: Learning Functional Dexterous Grasping from a Single Human Demo](../entities/paper-aff-2601-05243-generate-transfer-adapt-learning-functional-dexterous-gr.md) |
| IEEE Access 2026 | [One-Shot 3-D Affordance Learning for Multi-Stage Robotic Manipulation](../entities/paper-aff-one-shot-3-d-affordance-learning-for-multi-stage-robotic.md) |
| CVPR 2026 | [BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipato](../entities/paper-aff-2603-21679-bipremanip-learning-affordance-based-bimanual-preparator.md) |
| arXiv 2026 | [RAAP: Retrieval-Augmented Affordance Prediction with Cross-Image Action Alignment](../entities/paper-aff-2603-29419-raap-retrieval-augmented-affordance-prediction-with-cros.md) |
| arXiv 2026 | [BridgeACT: Bridging Human Demonstrations to Robot Actions via Unified Tool-Target Affordan](../entities/paper-aff-2604-23249-bridgeact-bridging-human-demonstrations-to-robot-actions.md) |
| arXiv 2026 | [Afford-VLA: Action-Aligned Visual Planning via Internalized Affordance](../entities/paper-aff-2605-24203-afford-vla-action-aligned-visual-planning-via-internaliz.md) |
| arXiv 2026 | [AffordVLA: Injecting Affordance Representations into Vision-Language-Action Models via Imp](../entities/paper-aff-2605-17517-affordvla-injecting-affordance-representations-into-visi.md) |
| CVPR 2026 | [PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Man](../entities/paper-aff-2601-07060-palm-progress-aware-policy-learning-via-affordance-reaso.md) |
| CVPR 2026 | [AffordGen: Generating Diverse Demonstrations for Generalizable Object Manipulation with Af](../entities/paper-aff-2604-10579-affordgen-generating-diverse-demonstrations-for-generali.md) |
| arXiv 2026 | [AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Afforda](../entities/paper-aff-2606-06155-affordancevla-a-vision-language-action-model-empowering.md) |
| arXiv 2026 | [Affordance2Action: Task-Conditioned Scene-level Affordance Grounding for Real-Time Manipul](../entities/paper-aff-2606-04172-affordance2action-task-conditioned-scene-level-affordanc.md) |
| arXiv 2026 | [GROW2: Grounding Which and Where for Robot Tool Use](../entities/paper-aff-2606-30632-grow2-grounding-which-and-where-for-robot-tool-use.md) |
| arXiv 2026 | [AffordSim: A Scalable Data Generator and Benchmark for Affordance-Aware Robotic Manipulati](../entities/paper-aff-2604-11674-affordsim-a-scalable-data-generator-and-benchmark-for-af.md) |
| arXiv 2026 | [A4A: Cross-Embodiment Transfer of Action-Oriented 4D Affordances from Human Demonstrations](../entities/paper-aff-2609-05892-a4a-cross-embodiment-transfer-of-action-oriented-4d-affo.md) |

### Datasets / Benchmarks

| 出处 | 论文 |
|------|------|
| ECCV 2018 | [Scaling egocentric vision: The epic-kitchens dataset](../entities/paper-aff-1804-02748-scaling-egocentric-vision-the-epic-kitchens-dataset.md) |
| CVPR 2020 | [Graspnet-1billion: A large-scale benchmark for general object grasping](../entities/painode-218-graspnet1billion.md) |
| CVPR 2022 | [Ego4d: Around the world in 3,000 hours of egocentric video](../entities/paper-ego4d.md) |

### Related Surveys

| 出处 | 论文 |
|------|------|
| IEEE TCDS 2016 | [Affordance Research in Developmental Robotics: A Survey](../entities/paper-aff-affordance-research-in-developmental-robotics-a-survey.md) |
| IEEE TCDS 2018 | [Affordances in Psychology, Neuroscience, and Robotics: A Survey](../entities/paper-aff-affordances-in-psychology-neuroscience-and-robotics-a-su.md) |
| IJCAI 2021 | [Building Affordance Relations for Robotic Agents - A Review](../entities/paper-aff-building-affordance-relations-for-robotic-agents-a-revie.md) |
| ACM Computing Surveys 2021 | [Visual Affordance and Function Understanding: A Survey](../entities/paper-aff-visual-affordance-and-function-understanding-a-survey.md) |
| IEEE TBD 2023 | [A Survey of Visual Affordance Recognition Based on Deep Learning](../entities/paper-aff-a-survey-of-visual-affordance-recognition-based-on-deep.md) |
| IEEE TCDS 2023 | [Recent Advances of Deep Robotic Affordance Learning: A Reinforcement Learning Perspective](../entities/paper-aff-recent-advances-of-deep-robotic-affordance-learning-a-re.md) |
| arXiv 2025 | [Visual Affordance Prediction: Survey and Reproducibility](../entities/paper-aff-2505-05074-visual-affordance-prediction-survey-and-reproducibility.md) |


## 局限与风险

- 索引级页面只含标题、出处与分类标签，**不替代**原文。
- 分组以条目首次出现的章节为准；跨分组条目的其他分组写在条目页内。
- 上游仍在更新，本页是 2026-10-23 的快照。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [affordance_awesome_catalog.md](../../sources/papers/affordance_awesome_catalog.md)
- [awesome-affordance-learning.md](../../sources/repos/awesome-affordance-learning.md)
- 上游：<https://github.com/hq-King/Awesome-Affordance-Learning>

## 推荐继续阅读

- [Awesome-Affordance-Learning GitHub](https://github.com/hq-King/Awesome-Affordance-Learning)
