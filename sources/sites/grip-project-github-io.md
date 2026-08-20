# GRIP 项目页（ryosukehori.github.io/grip-project）

> 来源归档（ingest 配套站点）

- **URL：** <https://ryosukehori.github.io/grip-project/>
- **对应论文：** [GRIP](https://arxiv.org/abs/2603.16233)（arXiv:2603.16233；CVPR 2026）
- **机构：** Carnegie Mellon University；Keio University / Keio AI Research Center
- **入库日期：** 2026-08-20
- **一句话说明：** 官方落地页：teaser 视频、方法四模块说明、PRISM 数据集画廊、三数据集定量/定性对比表与 BibTeX。

## 页面要点（2026-08 快照）

### 开放资源链接

| 资源 | URL |
|------|-----|
| 代码 | <https://github.com/RyosukeHori/GRIP> |
| 数据集 | <https://github.com/RyosukeHori/PRISM> |

### 方法模块（页内）

1. **Input Data** — 4 IMU + 鞋垫 GRF / CoP / 接触标签
2. **KinematicsNet** — 渐进 LSTM 估计叶关节、全身位置/角度、key 速度
3. **State Difference** — 估计态与仿真 humanoid 残差
4. **DynamicsNet** — PPO 物理 humanoid 控制 + fall recovery

### PRISM 速查

- **规模：** 1,275 条 × 10 s，6 被试，~3.5 h，100 Hz
- **模态：** IMU、鞋垫压力、光学 MoCap、3D 环境/物体模型
- **动作：** 日常 locomotion、慢速拉伸/深蹲、球类运动、踩物/坐物交互

## 对 wiki 的映射

- 与 [sources/papers/grip_arxiv_2603_16233.md](../papers/grip_arxiv_2603_16233.md)、[sources/repos/grip.md](../repos/grip.md) 配对
- 实体页：[wiki/entities/paper-grip.md](../../wiki/entities/paper-grip.md)
