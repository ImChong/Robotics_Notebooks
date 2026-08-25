# 近100余项工作调研！万字长文讲透具身世界模型的前世今生

> 来源归档（blog / 微信公众号）

- **标题：** 近100余项工作调研！万字长文讲透具身世界模型的前世今生——从动力学预测到机器人应用闭环
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ
- **发表日期：** 2026-08-25
- **入库日期：** 2026-08-25
- **抓取方式：** Agent Reach + `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_wm_six_routes_survey_2026-08-25/`](../raw/wechat_embodied_wm_six_routes_survey_2026-08-25/)
- **一句话说明：** 按 **模型构建 / 规划 / 学习 / 行动 / 评估 / 上下文** 六条闭环职责 taxonomy 串读具身世界模型百年脉络；核心判断：世界模型正从「生成未来内容」变为 **管理行动后果与决策不确定性的系统组件**。

## 核心摘录（归纳，非全文）

### 底层范式四环节

状态与记忆 → 动作条件转移 → 结果/奖励/风险 → 现实反馈与纠错。

### 六条技术路线（按闭环职责）

| 路线 | 使用时机 | 闭环作用 |
|------|----------|----------|
| 模型构建型 | 开发时 | 验证预测器本身 |
| 规划主导型 | 执行时 | 预测后果，外部裁决动作 |
| 学习主导型 | 训练时 | 想象经验训练策略 |
| 行动主导型 | 执行时 | 未来建模与动作生成耦合 |
| 评估主导型 | 部署前/测试时 | 外部策略在学习世界中考试 |
| 上下文主导型 | 持续运行 | 维护 World Proxy 状态与记忆 |

### 文内五个判断（2026-08）

1. 世界基础模型已出现但非通用物理引擎；可复用先验需后训练校准。
2. 评价从画质转向行动效用（WorldArena、RoboWM-Bench）。
3. 互联网视频扩观察规模，机器人数据补动作后果。
4. 短时可靠预测+持续纠错优于一次长开环生成。
5. 规划与策略融合但不互相取代；快慢双路可部署。

## 对 wiki 的映射

- **父节点**：[embodied-wm-six-routes-technology-map](../../wiki/overview/embodied-wm-six-routes-technology-map.md)
- **分类 hub**：`wiki/overview/embodied-wm-route-01-model-building.md` … `06-context.md` + `outlook`
- **论文实体**：文内点名 **56** 篇 → **38 复用既有节点 + 18 新建**（同一工作不重复造页）

### 论文索引

| 论文 | 路线 | arXiv | 节点状态 | wiki |
|------|------|-------|----------|------|
| ContactNets | 01 | [2011.08903](https://arxiv.org/abs/2011.08903) | 新建 | [paper-contactnets-contact-dynamics](../../wiki/entities/paper-contactnets-contact-dynamics.md) |
| GAIA-1 | 01 | [2309.17080](https://arxiv.org/abs/2309.17080) | 既有 | [paper-gaia1](../../wiki/entities/paper-gaia1.md) |
| Cosmos Predict | 01 | [2501.03575](https://arxiv.org/abs/2501.03575) | 既有 | [paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi](../../wiki/entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md) |
| Qwen-RobotWorld | 01 | — | 既有 | [paper-sa-2606-17030-qwen-robotworld-unifying-embodied-world-modeling](../../wiki/entities/paper-sa-2606-17030-qwen-robotworld-unifying-embodied-world-modeling.md) |
| Genie | 01 | [2402.15391](https://arxiv.org/abs/2402.15391) | 既有 | [paper-sa-2402-15391-genie-generative-interactive-environments](../../wiki/entities/paper-sa-2402-15391-genie-generative-interactive-environments.md) |
| Matrix-Game 3.5 | 01 | — | 既有 | [paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact](../../wiki/entities/paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact.md) |
| Visual Foresight | 02 | [1812.00568](https://arxiv.org/abs/1812.00568) | 新建 | [paper-visual-foresight-latent-mpc](../../wiki/entities/paper-visual-foresight-latent-mpc.md) |
| PETS | 02 | [1805.08034](https://arxiv.org/abs/1805.08034) | 新建 | [paper-pets-probabilistic-dynamics-mpc](../../wiki/entities/paper-pets-probabilistic-dynamics-mpc.md) |
| Resilient Machines (Self-Modeling) | 02 | [1903.00572](https://arxiv.org/abs/1903.00572) | 新建 | [paper-resilient-machines-continuous-self-modeling](../../wiki/entities/paper-resilient-machines-continuous-self-modeling.md) |
| PlaNet | 02 | [1811.09083](https://arxiv.org/abs/1811.09083) | 既有 | [paper-planet-latent-dynamics](../../wiki/entities/paper-planet-latent-dynamics.md) |
| MuZero | 02 | [1911.08265](https://arxiv.org/abs/1911.08265) | 新建 | [paper-muzero-planning-latent-dynamics](../../wiki/entities/paper-muzero-planning-latent-dynamics.md) |
| V-JEPA 2 | 02 | [2506.09985](https://arxiv.org/abs/2506.09985) | 既有 | [paper-vjepa2](../../wiki/entities/paper-vjepa2.md) |
| TD-MPC2 | 02 | — | 既有 | [paper-td-mpc2](../../wiki/entities/paper-td-mpc2.md) |
| DINO-WM | 02 | [2411.04983](https://arxiv.org/abs/2411.04983) | 既有 | [paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu](../../wiki/entities/paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md) |
| RoboCraft | 02 | [2205.02909](https://arxiv.org/abs/2205.02909) | 新建 | [paper-robocraft-particle-graph-dynamics](../../wiki/entities/paper-robocraft-particle-graph-dynamics.md) |
| ParticleFormer | 02 | — | 既有 | [paper-sa-2506-23126-particleformer-a-3d-point-cloud-world-model-for](../../wiki/entities/paper-sa-2506-23126-particleformer-a-3d-point-cloud-world-model-for.md) |
| PointWorld | 02 | — | 既有 | [paper-sa-2601-03782-pointworld](../../wiki/entities/paper-sa-2601-03782-pointworld.md) |
| τ₀-VLA | 02 | — | 既有 | [paper-tau0-vla](../../wiki/entities/paper-tau0-vla.md) |
| CheckVLA | 02 | [2607.26789](https://arxiv.org/abs/2607.26789) | 新建 | [paper-checkvla-execution-time-verification](../../wiki/entities/paper-checkvla-execution-time-verification.md) |
| World Models (Ha & Schmidhuber) | 03 | [1803.10122](https://arxiv.org/abs/1803.10122) | 既有 | [paper-ha-schmidhuber-world-models](../../wiki/entities/paper-ha-schmidhuber-world-models.md) |
| Dreamer | 03 | — | 既有 | [paper-dreamer-latent-imagination](../../wiki/entities/paper-dreamer-latent-imagination.md) |
| DreamerV3 | 03 | — | 既有 | [paper-shenlan-wm-13-dreamerv3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md) |
| RISE | 03 | — | 既有 | [paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio](../../wiki/entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md) |
| World4RL | 03 | — | 既有 | [paper-sa-2509-19080-world4rl-diffusion-world-models-for-policy-refin](../../wiki/entities/paper-sa-2509-19080-world4rl-diffusion-world-models-for-policy-refin.md) |
| Robotic World Model | 03 | — | 既有 | [robotic-world-model-eth-rsl](../../wiki/entities/robotic-world-model-eth-rsl.md) |
| DreamGen | 03 | — | 既有 | [paper-notebook-dreamgen-unlocking-generalization-in-robot-learn](../../wiki/entities/paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md) |
| DayDreamer | 03 | [2206.14176](https://arxiv.org/abs/2206.14176) | 新建 | [paper-daydreamer-world-models-real-robots](../../wiki/entities/paper-daydreamer-world-models-real-robots.md) |
| UniSim | 03 | — | 既有 | [paper-unisim](../../wiki/entities/paper-unisim.md) |
| GigaWorld-0 | 03 | — | 既有 | [gigaworld-0](../../wiki/entities/gigaworld-0.md) |
| GR00T-Dreams | 03 | — | 既有 | [paper-gr00t-dreams-synthetic-trajectories](../../wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md) |
| Unified World Models (UWM) | 04 | — | 既有 | [paper-shenlan-wm-08-uwm](../../wiki/entities/paper-shenlan-wm-08-uwm.md) |
| Cosmos Policy | 04 | — | 既有 | [paper-shenlan-wm-11-cosmos-policy](../../wiki/entities/paper-shenlan-wm-11-cosmos-policy.md) |
| DreamZero | 04 | — | 既有 | [paper-notebook-dreamzero-world-action-models-are-zero-shot-poli](../../wiki/entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md) |
| Riemann-1.0 | 04 | — | 新建 | [paper-riemann-1-causal-action-video-wam](../../wiki/entities/paper-riemann-1-causal-action-video-wam.md) |
| World Tokens | 04 | — | 新建 | [paper-world-tokens-inference-trimmed-wam](../../wiki/entities/paper-world-tokens-inference-trimmed-wam.md) |
| FLEX-π | 04 | [2608.10860](https://arxiv.org/abs/2608.10860) | 既有 | [paper-flex-pi](../../wiki/entities/paper-flex-pi.md) |
| MobileWAM | 04 | — | 新建 | [paper-mobilewam-mobile-manipulation-wam](../../wiki/entities/paper-mobilewam-mobile-manipulation-wam.md) |
| MotionWAM | 04 | [2606.09215](https://arxiv.org/abs/2606.09215) | 既有 | [paper-motionwam-humanoid-loco-manipulation-wam](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) |
| WorldGym | 05 | [2506.00613](https://arxiv.org/abs/2506.00613) | 既有 | [paper-shenlan-wm-15-worldgym](../../wiki/entities/paper-shenlan-wm-15-worldgym.md) |
| Veo World Simulator | 05 | — | 新建 | [paper-veo-world-simulator-policy-testing](../../wiki/entities/paper-veo-world-simulator-policy-testing.md) |
| GE-Sim 2.0 | 05 | [2605.27491](https://arxiv.org/abs/2605.27491) | 既有 | [ge-sim-2](../../wiki/entities/ge-sim-2.md) |
| WorldEval | 05 | [2505.19017](https://arxiv.org/abs/2505.19017) | 既有 | [paper-sa-2505-19017-worldeval-world-model-as-real-world-robot-polici](../../wiki/entities/paper-sa-2505-19017-worldeval-world-model-as-real-world-robot-polici.md) |
| GigaWorld-1 | 05 | — | 既有 | [paper-gigaworld-1-policy-evaluation](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md) |
| Hydra | 06 | — | 既有 | [paper-hydra-0](../../wiki/entities/paper-hydra-0.md) |
| ConceptGraphs | 06 | [2309.16650](https://arxiv.org/abs/2309.16650) | 新建 | [paper-conceptgraphs-open-vocabulary-3d-scene](../../wiki/entities/paper-conceptgraphs-open-vocabulary-3d-scene.md) |
| HoloAgent-0 | 06 | — | 既有 | [holoagent](../../wiki/entities/holoagent.md) |
| SayPlan | 06 | [2307.01871](https://arxiv.org/abs/2307.01871) | 新建 | [paper-sayplan-llm-scene-graph-planning](../../wiki/entities/paper-sayplan-llm-scene-graph-planning.md) |
| RoboMemory | 06 | — | 新建 | [paper-robomemory-multi-type-embodied-memory](../../wiki/entities/paper-robomemory-multi-type-embodied-memory.md) |
| Cosmos 3 | future | — | 既有 | [cosmos-3](../../wiki/entities/cosmos-3.md) |
| WorldArena | future | [2602.08971](https://arxiv.org/abs/2602.08971) | 既有 | [paper-sa-2602-08971-worldarena-a-unified-benchmark-for-evaluating-pe](../../wiki/entities/paper-sa-2602-08971-worldarena-a-unified-benchmark-for-evaluating-pe.md) |
| RoboWM-Bench | future | — | 新建 | [paper-robowm-bench-action-faithfulness](../../wiki/entities/paper-robowm-bench-action-faithfulness.md) |
| DreamDojo | future | — | 既有 | [paper-hrl-stack-35-dreamdojo](../../wiki/entities/paper-hrl-stack-35-dreamdojo.md) |
| PlayWorld | future | — | 新建 | [paper-playworld-autonomous-play-data](../../wiki/entities/paper-playworld-autonomous-play-data.md) |
| Newton | future | — | 既有 | [newton-physics](../../wiki/entities/newton-physics.md) |
| PIN-WM | future | [2504.16693](https://arxiv.org/abs/2504.16693) | 既有 | [paper-sa-2504-16693-pin-wm-learning-physics-informed-world-models-fo](../../wiki/entities/paper-sa-2504-16693-pin-wm-learning-physics-informed-world-models-fo.md) |
| Foresight (PI) | future | — | 新建 | [paper-foresight-action-conditioned-failure-monitoring](../../wiki/entities/paper-foresight-action-conditioned-failure-monitoring.md) |

## 当前提炼状态

- [x] 正文抓取与归纳摘要
- [x] 文内点名论文独立节点（复用优先）
- [x] 六路线分类 hub + 父技术地图
