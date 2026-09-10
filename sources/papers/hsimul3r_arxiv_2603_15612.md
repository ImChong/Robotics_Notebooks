# HSImul3R: Physics-in-the-Loop Reconstruction of Simulation-Ready Human–Scene Interactions（arXiv:2603.15612）

> 来源归档（ingest）

- **标题：** HSImul3R: Physics-in-the-Loop Reconstruction of Simulation-Ready Human–Scene Interactions
- **类型：** paper / human-scene-interaction / reconstruction / physics-in-the-loop / sim2real / humanoid
- **arXiv abs：** <https://arxiv.org/abs/2603.15612>
- **PDF：** <https://arxiv.org/pdf/2603.15612>
- **Hugging Face Papers：** <https://huggingface.co/papers/2603.15612>
- **项目页：** <https://yukangcao.github.io/HSImul3R/>
- **代码：** <https://github.com/yukangcao/HSImul3R>（截至入库日仅 README + 静态 docs，**无可运行训练/推理入口**）
- **机构：** 南洋理工大学 S-Lab（NTU）、上海人工智能实验室（Shanghai AI Lab）、大晓机器人（ACE Robotics）
- **入库日期：** 2026-09-10
- **一句话说明：** 从稀疏视角图像或单目视频做 **simulation-ready** 人–场景交互 3D 重建；用 **物理仿真双向优化**（正向 scene-targeted RL  refine 人体运动、反向 DSRO refine 场景几何），并发布 **HSIBench** 16 视角同步采集基准；优化后人体运动可迁移到人形机器人。

## 摘要级要点

- **问题：** 现有 HSI 重建 **感知–仿真鸿沟**——视觉 plausible 但违反物理约束 → 物理引擎不稳定、具身 AI 难用。
- **输入：** casual captures（稀疏视角图像、单目视频）。
- **先验：** 注入 **3D explicit generative prior**，改善人体与场景对齐。
- **正向 pass（人体）：** **Scene-targeted Reinforcement Learning**，在仿真中优化人体运动，双重监督 **motion fidelity + contact stability**。
- **反向 pass（场景）：** **Direct Simulation Reward Optimization (DSRO)**，用仿真反馈（重力稳定、交互成功） refine 场景几何。
- **DSRO 四类反馈：** Type 1 物体未在重力下稳定；Type 2 交互中不稳定；Type 3 稳定但无有效交互；Type 4 稳定且有有效交互。
- **HSIBench：** 16-view 同步采集、多样物体/受试者/动作的人–场景交互数据集。
- **下游：** 项目页展示优化后人体运动 **可无缝迁移部署到人形机器人**（与 [PhysHSI](../../wiki/entities/paper-amp-survey-15-physhsi.md) 等 **控制侧 HSI** 形成 reconstruction → retarget → control 链路）。

## 核心摘录（面向 wiki 编译）

### 与相邻 HSI 路线对照

| 维度 | HSImul3R（本文） | DIMOS | PhysHSI | COINS |
|------|------------------|-------|---------|-------|
| 目标 | **Simulation-ready 3D 重建** | SMPL 运动 **合成** | G1 **真机 HSI 控制** | SMPL-X **静态姿态合成** |
| 物理 | **仿真器作 active supervisor** | RL 任务奖励 | AMP + onboard 感知 | 几何/语义约束 |
| 输入 | 稀疏图 / 单目视频 | 场景 + 任务 | MoCap retarget + 仿真 | 场景 + action–object 语义 |
| 输出 | 可进物理引擎的 HSI | 运动轨迹 | 29-D 策略 | 静态交互姿态 |

### 开源核查（2026-09-10）

- **项目页：** Paper / Webpage 外链；**未在首页显式列出 GitHub**。
- **GitHub：** 仓库已建（2026-03-16），含 README、BibTeX、`docs/` 静态页；**无** `requirements.txt`、训练/推理脚本或权重。
- **结论：** **部分开源（占位仓库）** — 可引用与跟踪发布，**暂不可复现**。

## 对 wiki 的映射

- 沉淀实体页：[HSImul3R](../../wiki/entities/paper-hsimul3r.md)
- 项目页归档：[sources/sites/hsimul3r-github-io.md](../sites/hsimul3r-github-io.md)
- 仓库归档：[sources/repos/hsimul3r.md](../repos/hsimul3r.md)
- 交叉：[DIMOS](../../wiki/entities/paper-dimos-human-scene-motion-synthesis.md)、[PhysHSI](../../wiki/entities/paper-amp-survey-15-physhsi.md)、[TokenHSI](../../wiki/entities/paper-bfm-38-tokenhsi.md)、[COINS](../../wiki/entities/paper-coins-compositional-human-scene-interaction.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)

## 参考来源（原始）

- arXiv:2603.15612
- [HSImul3R 项目页](https://yukangcao.github.io/HSImul3R/)
- [GitHub: yukangcao/HSImul3R](https://github.com/yukangcao/HSImul3R)
