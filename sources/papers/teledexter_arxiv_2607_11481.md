# Towards Human-level Dexterous Teleoperation（TeleDexter）

> 来源归档（ingest）

- **标题：** Towards Human-level Dexterous Teleoperation
- **简称：** TeleDexter
- **类型：** paper / dexterous-teleoperation / hand-object-co-tracking / sim2real / imitation-learning
- **arXiv abs：** <https://arxiv.org/abs/2607.11481>
- **PDF（项目页）：** <https://bigai-dex.github.io/blog/teledexter/paper_teledexter.pdf>
- **提交日期：** 2026-07-13
- **项目页：** <https://bigai-dex.github.io/blog/teledexter/>
- **机构：** 清华大学、北京通用人工智能研究院（BIGAI）、北京大学
- **作者：** Puhao Li\*、Zeyuan Chen\*、Yingying Wu\*、Pengkun Wei、Yuyang Li、Tianyu Wang、Jiaxiao Shi、Mingrui Yu、Baoxiong Jia、Song-Chun Zhu、Tengyu Liu†、Siyuan Huang†（\* 共同一作，† 通讯）
- **仿真 / 训练：** Isaac Gym；SAPG；约 62k 并行环境、4× RTX 5090；单物体参考运动约 50 分钟
- **真机：** Franka FR3 + SharpaWave（22 DoF）/ LeapHand（16 DoF）；NOKOV MoCap 30 Hz
- **开源状态：** **未开源**（截至 2026-07-28：项目页未列代码；GitHub 无 `teledexter` 公开仓）
- **入库日期：** 2026-07-28
- **一句话说明：** 提出 hand–object **co-tracking** 低层控制器：操作员给出同步指尖位姿与物体位姿目标，仿真单阶段 RL（连续子目标 + hybrid reward + random action masking）零样本部署，七项灵巧遥操作平均 **75.2% SR**，并可采数训 Diffusion Policy。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://bigai-dex.github.io/blog/teledexter/> | 演示、表格、局限与引用 |
| 项目页归档 | [teledexter-project.md](../sites/teledexter-project.md) | 开源核查与摘录 |
| 运动学重定向基线工具 | [dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 文中 DexRT 所用开源向量对齐重定向 |
| GeoRT | <https://github.com/facebookresearch/GeoRT> | 学习式几何重定向基线 |
| SimToolReal | <https://github.com/tylerlum/simtoolreal> | 物体中心工具使用对照（非遥操作） |
| 同主题 co-tracking（人形全身） | [HDMI](../../wiki/entities/paper-hrl-stack-06-hdmi.md) | robot–object co-tracking 对照（loco-manip，非灵巧手） |

## 摘要级要点

- **问题：** 现有灵巧遥操作多为关节运动学镜像，丢弃手–物接触力与惯性；finger gaiting / 手内重定向 / 工具功能抓切换时易滑落。生成式动作先验（DexGen）又易在长程闭环中漂移。
- **方法：** 将遥操作表述为 **hand–object co-tracking**：目标 \(g_t=(\hat p^{\mathrm{tip}}_t,\hat T^o_t)\)；策略输出关节目标。人类 MoCap HOI → 两阶段几何感知重定向 → 连续子目标 RL（sparse 到达 + dense 跟踪 + 课程 + **random action masking**）。
- **评测：** SharpaWave 七任务平均 **75.2% SR / 87.1% TP**（每任务 15 trials）；DexRT 5.7%、GeoRT/DexGen ≈0%。LeapHand 重定向任务 **60–73.3% SR**（同源人类参考，仅改重定向）。
- **数据飞轮：** 每任务 50 条示范训 Conv-UNet Diffusion Policy：HammerDriver **73.3%**、BulbInstall **46.7%**、BrushForward **40.0%**。
- **消融：** 连续子目标 ≫ 逐帧 dense tracking（仿真 EpLen / Goals）；去掉 action masking 真机 SR 大幅下降（如 ScrewdriverUse 73.3%→0%）。
- **局限：** object-specific；依赖 MoCap；失败模式含工具–环境冲击扰动、接触切换卡滞、无触觉导致的 tracking stall。

## 核心摘录（面向 wiki 编译）

### 1) 连续子目标 co-tracking

参考轨迹按可变间隔采样同步指尖 + 物体位姿子目标；策略须按序到达，子目标之间自由发现接触策略。到达判据：连续 \(N_{\mathrm{stay}}\) 帧内指尖/物体位置/旋转误差均小于阈值。

### 2) Hybrid reward

\[
r_t = \mathbf{1}_{\mathrm{reach}}(t)\, w_{\mathrm{step}}(t)\, r_{\mathrm{score}}(t) + \alpha_{\mathrm{dense}} r_{\mathrm{dense}}(t) - c_{\mathrm{time}}
\]

稀疏到达奖励按子目标跨度加权；dense 项提供早期 shaping。

### 3) 课程与 Sim2Real

重力从弱到满；跟踪容差由松到紧；子目标间距由短到长。域随机（手/物体形状与动力学、外力、观测噪声与延迟）+ **random action masking**（随机冻结关节维并保持旧指令）为最关键零样本转移技巧。

### 4) 几何感知重定向

阶段 1：向量对齐运动学重定向；阶段 2：物体 mesh SDF 表面吸引 + 穿透惩罚 + 指间碰撞球 + cuRobo 时序平滑。

### 5) 部署协议

预抓取用运动学重定向；稳定接触后切换到 co-tracking；臂部独立 IK 跟踪腕部。

## 对 wiki 的映射

- [TeleDexter 论文实体](../../wiki/entities/paper-teledexter.md)
- [Teleoperation](../../wiki/tasks/teleoperation.md)
- [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
- [In-hand Reorientation](../../wiki/methods/in-hand-reorientation.md)
- [灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md)
- [HDMI（robot–object co-tracking 对照）](../../wiki/entities/paper-hrl-stack-06-hdmi.md)
- [深度遥操作路线](../../roadmap/depth-teleoperation.md)

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2607.11481>
- 项目页：<https://bigai-dex.github.io/blog/teledexter/>
- PDF：<https://bigai-dex.github.io/blog/teledexter/paper_teledexter.pdf>
