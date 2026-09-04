# The Imitator Game（意图级模仿基准）

> 来源归档（ingest）

- **标题：** The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.22301>
- **机构：** 香港大学（HKU）；超忆（TranscEngram）；复旦大学（Fudan）；浙江大学（ZJU）
- **作者：** Xunzhe Zhou（项目负责）、Yiyang Cai、Fengyi Wang、Ran Ju（以上共同一作）、Hanxiang Ren、Ruizhe Liu、Yu Zhang、Qian Luo、Feng Chen、Pei Zhou、Yi Ma、Yanchao Yang（通讯）
- **项目页：** <https://imitator-game.github.io/> — 归档见 [`sources/sites/imitator-game-github-io.md`](../sites/imitator-game-github-io.md)
- **代码：** <https://github.com/imitator-game/The-Imitator-Game>（MIT）— 归档见 [`sources/repos/the-imitator-game.md`](../repos/the-imitator-game.md)
- **数据：** <https://huggingface.co/datasets/imitator-game/IG-10K-Dataset> — 归档见 [`sources/datasets/ig-10k.md`](../datasets/ig-10k.md)
- **入库日期：** 2026-08-30
- **再核日期：** 2026-09-02
- **一句话说明：** L0–L3 四级基准把人类示范与机器人现场的差距逐步拉大，用目标等价而非动作相似衡量模仿；配套 IG-10K 与盲测 Arena。

## 核心摘录

### 1) 轨迹复现 ≠ 意图模仿

- **摘录要点：** 现有策略学 \(\pi(a_t\mid o_t,V)\) 的 observation→action，面对人类视频多在近似场景复现轨迹。真正的模仿是推断目标，并用手头工具/物体/布局完成同一意图。策略**不**接收任务描述或层级标签；VLA 经平台侧固定 captioner \(T(V)\) 才看到语言。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [模仿学习](../../wiki/methods/imitation-learning.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 场景分解与四级保真

- **摘录要点：** 场景 \(\mathcal{S}=(\{O_i=(A_i,G_i,S_i)\},P)\)。L0 保 \(P,A,G,S\)（轨迹匹配）；L1 改 \(P\)（物体终态）；L2 改 \(A/G\)、保语义 \(S\)（语义任务）；L3 连 \(S\) 也换（功能替代 / affordance 适应）。同一人类 clip 服务四级，只改机器人侧场景。具身差距大致固定，与 RHyME 等 embodiment-gap 层级正交。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 3) IG-10K + Arena 协议

- **摘录要点：** 20,000+ 配对（真机 11.7K Realman VR 遥操作 + 仿真 10K 双臂 Franka / ManiSkill3 OMPL），50+ 任务 × 四级、6 领域，LeRobot-0.5.0；多视角 + MANO / 分割 / 三层语言。主评测固定 5 seen + 5 unseen × L0–L3。仿真报 SR / Sub-SR；Arena 盲测 SR_human / \(Q\) / WR（仿真 15k、真机 5k pairwise）。自动 SR 与人类判断 \(r\approx 0.858/0.861\)。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [具身评测基准枢纽](../../wiki/overview/hub-embodied-eval-benchmark.md)
  - [LeRobot](../../wiki/entities/lerobot.md)

### 4) 九模型横评数字

- **摘录要点：** 冻结视觉/VLM 骨干，只训动作头。仿真 seen：ACT/DINOv2 SR **0.81**，XSkill **0.79**，\(\pi_{0.5}\) **0.73**。未见任务零样本全部 **<13%**。P+FT（10 条配对）多数优于从零 few-shot，且随预训练任务数 15→45 增大。真机 P+FT：L0–L2 均值 SR≈0.40，L3 掉到 **0.29**；XSkill 在 L0–L2 最强（≈0.53–0.57）但 L3 落到 0.29。人视频条件优于字幕条件；DINOv2 / SigLIP2 优于 VideoMAE。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [模仿学习](../../wiki/methods/imitation-learning.md)

### 5) 开源状态（截至 2026-09-04 再核仍成立）

- **摘录要点：** **已开源（MIT）**。官方仓含 ManiSkill/SAPIEN 仿真、采集与九套基线训练/评测入口；HF 发布 IG-10K-Dataset 与 IG-10K-Assets（ModelScope 镜像 `Zhouxunzhe/*`）。Arena / 文档 / 数据页在项目站。真机部署评测走社区申请，仓内无上传接口。2026-08-30 浅入库时训练仓尚未公开，本次复核已改判。

## 当前提炼状态

- [x] 项目页与 arXiv 全文对齐（2026-09-02 再核）
- [x] 步骤 2.5：GitHub + HF 数据/资产已发布
- [x] wiki 映射：`wiki/entities/paper-imitator-game.md` 深化
