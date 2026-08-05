# Teleopit: A Full-Embodiment Humanoid Teleoperation System（arXiv:2608.01834）

> 来源归档（ingest）

- **标题：** Teleopit: A Full-Embodiment Humanoid Teleoperation System
- **类型：** paper / humanoid / teleoperation / whole-body tracking / dexterous hand / VR / active vision
- **arXiv abs：** <https://arxiv.org/abs/2608.01834>
- **PDF：** <https://arxiv.org/pdf/2608.01834>
- **项目页：** <https://botrunner64.github.io/teleopit-page/>
- **代码（主仓）：** <https://github.com/BotRunner64/Teleopit>
- **配套仓：** [somehand](https://github.com/BotRunner64/somehand)、[pico-bridge](https://github.com/BotRunner64/pico-bridge)、[OpenNeck](https://github.com/BotRunner64/OpenNeck)、[lerobot-teleopit](https://github.com/BotRunner64/lerobot-teleopit)
- **机构：** 西湖大学（Westlake University）、上海创智学院（Shanghai Innovation Institute）；通讯 Xiangru Huang
- **作者：** Bingqian Wu、Zicheng Xu、Xianghui Fan、Dayu Li、Xiangru Huang
- **发表 / 上传：** 2026-08-03（arXiv）
- **硬件：** Unitree G1（29 DoF）；多款商业灵巧手；OpenNeck 2-DoF 主动颈
- **仿真栈：** **mjlab**（GPU MuJoCo）；PPO；策略 50 Hz / PD 200 Hz
- **入库日期：** 2026-08-05
- **一句话说明：** 用 **PICO VR** 统一提供身体 / 手 / 头意图，组合 **History Encoder + failure-aware rewind** 的全身跟踪、**归一化指向量 + 指尖距离/拇指帧** 的跨手优化重定向，以及主动视觉与异步录制；96 条演示上 ACT/GR00T 达 90%/95% SR。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [teleopit-page](https://botrunner64.github.io/teleopit-page/) | 演示、Latency、五仓入口 |
| 文档 | [BotRunner64.github.io/Teleopit](https://BotRunner64.github.io/Teleopit/) | 安装、教程、架构（含中文） |
| 主仓 | [Teleopit](https://github.com/BotRunner64/Teleopit) | 训练 / sim2sim / sim2real / 录制 |
| 手重定向 | [somehand](https://github.com/BotRunner64/somehand) | 跨形态灵巧手优化目标 |
| 对照系统 | [TWIST2](https://github.com/amazon-far/TWIST2)、[HEFT](https://heft.axell.top/) | 便携 VR 全身 / 重载 VR 跟踪 |
| 下游引用 | [OASIS](../../wiki/entities/paper-loco-manip-04-oasis.md)、[MimicLite](../../wiki/entities/mimiclite.md) | 低层 WBC / 跨 codebase 部署列表 |

## 摘要级要点

- **问题：** 现有 VR 人形系统常只控上身/手，或全身但手退化为离散夹爪；连续全身+灵巧手又依赖定制惯性衣/手套；灵巧手重定向常需手型相关几何缩放或目标手训练数据。
- **Teleopit 回答：** VR 为统一意图源 → 全身 tracker + 优化手 retargeter + 主动视觉 + 异步 runtime；录制演示可直接喂 ACT / GR00T。
- **跟踪：** mjlab + PPO；History Encoder（H=10，Conv1d+GAP）；failure-aware rewind sampling；actor 可部署观测 vs critic 特权态。
- **手：** 归一化指方向目标去掉骨长尺度；指尖距离与拇指帧目标保 pinch / 对掌；新手只需语义 link 映射，共享权重与求解器设置。
- **开源状态（截至入库）：** 五仓 **已开源**（见项目页）；主仓含 G1 匹配 checkpoint（`ckpt/track_g1*.onnx`）。

## 核心摘录（面向 wiki 编译）

### 1) 系统 I/O

| 通道 | 输入（PICO） | 输出 |
|------|--------------|------|
| 身体 | 24-joint skeleton \(\mathbf{s}^{\mathrm{body}}\) | 29-D 身体关节目标 \(\mathbf{a}^{\mathrm{body}}\) |
| 手 | 每手 26 keypoints \(\mathbf{s}^{\mathrm{hand}}\) | 灵巧手命令 \(\mathbf{a}^{\mathrm{hand}}\) |
| 头 | head pose \(\mathbf{s}^{\mathrm{head}}\) | 2-DoF 视点 \(\mathbf{a}^{\mathrm{cam}}\) |

关节目标：\(\mathbf{q}^{\mathrm{target}}=\mathrm{clip}(\mathbf{a},-c,c)\odot\mathbf{s}+\mathbf{q}^{\mathrm{def}}\)；PD 200 Hz，策略 50 Hz；躯干锚定，跟踪 14 个身体 link。

### 2) 训练与评测数据（Table 4）

| 集合 | 来源 | 规模（约） |
|------|------|------------|
| Train | BONES-SEED / TWIST2 / LAFAN1 子集 + 9 段 PICO | ~220 h 级 mocap 片段 + 31 min PICO |
| Val mocap | BONES-SEED 子集 | 238 clips → 181×10 s windows |
| Val PICO | 系统自采 | 5 recordings → 67×10 s windows |

训练：8×A800，8192 env/GPU（共 65536），约 **50 h** wall time。

### 3) 跟踪对比（Table 6，持出集）

| Method | Mocap SR↑ | PICO SR↑ |
|--------|-----------|----------|
| TWIST2 | 43.1% | 64.2% |
| SONIC | 75.7% | 82.1% |
| HoloMotion | 64.6% | 97.0% |
| **Teleopit** | **91.7%** | **100.0%** |

消融（Table 7，降配 4 GPU / 20k iter）：Full (reduced) SR 74.0%；w/o rewind 72.9%；w/o history 73.5%。

### 4) 采数 → 自主（Table 13，瓶放置）

| Control | 训练集 | Success / trials | SR |
|---------|--------|------------------|-----|
| Teleoperation | — | 96 / 100 | 96.0% |
| ACT | 96 | 18 / 20 | 90.0% |
| GR00T N1.7 | 96 | 19 / 20 | 95.0% |

### 5) 与仓库内路线对照

| 维度 | Teleopit | TWIST2 | HEFT | TeleGate |
|------|----------|--------|------|----------|
| 传感 | **PICO VR 身体+手+头** | PICO + 2-DoF 颈 | raw VR（重噪声） | 惯性动捕 |
| 手 | **连续跨形态优化重定向** | 多为离散夹爪 | 非主叙事 | 非主叙事 |
| 跟踪创新 | History + rewind | 便携采集 + visuomotor | PMG + WPC 重载 | 门控专家 + VAE |
| 机体 | G1 | G1 | G1 + L7 | G1 |
| 下游 | ACT / GR00T on 96 demos | 扩散 visuomotor | 重载遥操作 | 高动态跟踪 |

## 对 wiki 的映射

- 沉淀实体页：[Teleopit（论文实体）](../../wiki/entities/paper-teleopit.md)
- 项目页：[teleopit-project.md](../sites/teleopit-project.md)
- 代码：[teleopit.md](../repos/teleopit.md)、[somehand.md](../repos/somehand.md)
- 交叉补强：[Teleoperation](../../wiki/tasks/teleoperation.md)、[TWIST2](../../wiki/entities/paper-twist2.md)、[HEFT](../../wiki/entities/paper-heft.md)、[OASIS](../../wiki/entities/paper-loco-manip-04-oasis.md)、[MimicLite](../../wiki/entities/mimiclite.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)

## 当前提炼状态

- [x] 项目页五仓开源核查（步骤 2.5）
- [x] arXiv PDF 方法 / Table 6–7 / Table 13 / 时延摘录
- [x] 主仓 README 与文档入口对齐
- [x] wiki 实体页与 teleoperation / OASIS / MimicLite 交叉链接规划
