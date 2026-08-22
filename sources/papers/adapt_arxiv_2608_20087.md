# AdaPT（arXiv:2608.20087）

> 来源归档（ingest）

- **标题：** Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking（**AdaPT**）
- **类型：** paper / humanoid / motion-imitation / ball-sports
- **arXiv：** <https://arxiv.org/abs/2608.20087>
- **项目页：** <https://humanoidtennis.github.io/AdaPT/>
- **代码：** <https://github.com/noitom-robotics/AdaPT>（Apache-2.0）
- **机构：** 诺亦腾机器人（Noitom Robotics）；上海人工智能实验室；上海交通大学；越疆机器人（Dobot Robotics）
- **入库日期：** 2026-08-22
- **索引来源：** [具身智能小站 10 篇盘点](../blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)（<https://mp.weixin.qq.com/s/EmC4gNgcQdPX34vxy-qSVQ>）
- **一句话说明：** 人形网球 **自适应规划–跟踪** 框架：从转播视频/MoCap 提取职业球员风格，分层 MVAE 规划 + 速度自适应跟踪；G1 与 Atom P3 真机对拉/发球，野外发球验证。

## 开源状态（步骤 2.5，2026-08-22）

| 资源 | 状态 |
|------|------|
| 项目页 | **已发布**（演示视频、数据集统计、GLB 预览） |
| GitHub | **部分开源** — Stage1 发球速度自适应跟踪训练/推理（`uv run train/play`）；对拉规划与完整部署管线待后续 release |
| 预训练权重 | `ckpts/player1/model_24000.pt` 等示例 checkpoint 已随仓发布 |

## 核心论文摘录

### 1) 问题与总贡献（Abstract / §1）

- **核心贡献：** 现有球类人形工作偏重 **击球成功率**，较少保留 **职业运动风格**（全身协调、发力与恢复）。AdaPT 提出 **Adaptive motion Planning and Tracking**：规划器生成 **风格化运动学轨迹**，跟踪器以最小干扰执行；通过 **随机化执行速度训练** 与 **规划器速度适配变量 \(\alpha\)** 缓解 sim-to-real 中跟踪退化、自回归规划误差与感知噪声的复合漂移。
- **对 wiki 的映射：**
  - [AdaPT 论文实体](../../wiki/entities/paper-adapt.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)

### 2) 数据与风格化管线（§3.1）

- **核心贡献：** 从 Nadal / Federer / Djokovic 转播片段（约 2 s/clip）经 **GVHMR → GMR 重定向** 得 SMPL/机器人运动；MoCap 运动员 **Mr. Black** 补充风格。标注 stroke / spin / 触球时刻；发球额外标 **抛球释放时刻**。对拉数据经 **通用跟踪器物理修正** 得 \(\mathcal{D}_{\mathrm{rally}}\)、\(\mathcal{D}_{\mathrm{serve}}\)。
- **对 wiki 的映射：**
  - [AdaPT 项目页](../sites/adapt-humanoidtennis.md)
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md)（对照：统一全身跟踪栈）

### 3) 速度自适应跟踪与分层规划（§3.2–3.3）

- **跟踪：** 参考与当前构型差 + 本体速度观测；训练时 \(\hat{q}_t^\alpha=(1-\alpha)\hat{q}_{t-1}+\alpha\hat{q}_t\)，\(\alpha\sim\mathrm{Unif.}(\alpha_{\min},\alpha_{\max})\) 暴露变速执行。
- **对拉：** 基于 Vid2Player3D 的 **MVAE 运动生成器** + 高层规划 \(\pi^{\mathrm{rally}}_{\mathrm{plan}}\) 输出 \((z_t,\alpha_t)\)，条件于机器人姿态与未来球轨迹估计。
- **发球：** 基于 AdaMimic 的 **残差跟踪器** \(\Delta a_t\) 适应抛球变化；关键帧（最深引拍）稀疏局部跟踪奖励 + 抛物线抛球引导。
- **对 wiki 的映射：**
  - [AdaPT 官方仓](../repos/adapt.md)

### 4) 实验与真机（§3.4–3.5 / Table 2）

- **仿真：** Mjlab + PPO，4096 并行 env，4×RTX 4090；对拉/发球相对 RL-Scratch、AMP、PULSE、Vid2Player3D、AdaMimic 等在 **成功率与风格保真** 上更均衡。
- **真机：** **Unitree G1**、**Dobot Atom（~1.7 m）**；野外发球用 YOLO 球检测 + HTC VIVE 定位，无 MoCap。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)（体育竞技子类）

### 5) 局限与谱系

- **局限：** 当前开源主要为 **Stage1 发球跟踪**；对拉 MVAE 规划与完整 sim2real 栈未全量发布；风格数据依赖特定球员与转播视角质量。
- **谱系：** 延续 Vid2Player3D（解耦规划–跟踪）、AdaMimic（速度自适应模仿）、LATENT（人形网球任务性能线）对照轴。
- **对 wiki 的映射：**
  - [Behavior Cloning](../../wiki/methods/behavior-cloning.md)
