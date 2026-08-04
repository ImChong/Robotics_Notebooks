# hifi_umi_arxiv_2607_25895

> 来源归档（ingest）

- **标题：** HiFi-UMI: Learning Deployable Manipulation Policies from High-Fidelity UMI Data Alone
- **类型：** paper
- **来源：** arXiv:2607.25895（2026-07-28）
- **作者：** Simple AI（Yuteng Wei, Jinming Ma, Jiawei Wang 等）
- **机构：** 简易人工智能（Simple AI）
- **入库日期：** 2026-07-30
- **最后更新：** 2026-08-04
- **项目页：** <https://cloud.simpleai.tech/simple-world-lab/hifi-umi/>
- **数据集：** <https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K>（CC BY 4.0；**已公开**）
- **一句话说明：** 高保真无机器人双臂 UMI 采数系统（约 3 mm 轨迹精度、<40 µs 同步、六视角）；证明仅用 HiFi-UMI 后训练即可匹配同域真机遥操作；开源 2000 小时 HiFi-UMI-2K。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract）

- **链接：** <https://arxiv.org/abs/2607.25895>
- **核心贡献：** 可部署操作策略受困于「既高保真又可规模化」数据稀缺：真机遥操作准但难扩；robot-free UMI 易扩，现有实践多用于预训练、后训练仍需少量真机 **anchor**。HiFi-UMI 共设计硬件/软件以提升轨迹精度、双夹爪相对位姿、同步与视场（头戴离线 stereo-inertial SLAM、**原生**相对位姿、共享 µs 级 GPIO 触发、每手两路广角约 **200°**），无外置追踪即可达约 **3 mm** 工作空间局部 EE 精度。展示 **zero-robot post-training**：仅在 HiFi-UMI 上后训练的策略可直接上真机，并在 StarVLA-QwenPI / OpenPI-π₀.₅ / LingBot-VA 上与同域遥操作成功率差 **-2.5 / +3.1 / -0.6 pp**；最强策略精密插入达 **85%**。同语料 **4000 h** 预训练使 10 个未见任务动作误差降 **41%**，StarVLA-QwenPI 真机再升 **+18.1 pp**。开源 **HiFi-UMI-2K**（2000 h，µs 同步、超宽 FoV，仿真回放校验）。
- **对 wiki 的映射：**
  - [HiFi-UMI 论文实体](../../wiki/entities/paper-hifi-umi.md)
  - [HandUMI](../../wiki/entities/handumi.md)
  - [遥操作](../../wiki/tasks/teleoperation.md)

### 2) 采数装置与数据引擎（§3）

- **采集：** 头戴 stereo+IMU 离线 SLAM；双手 marker 在同一头相机坐标系 → 原生双手相对位姿；GPIO 硬同步；六路相机（头立体 + 每手两路鱼眼）。
- **数据引擎：** 上传 → 轨迹重建与自动清洗 → **仿真 WBC 回放校验** → AI 辅助标注 → 人工抽检 → 导出。源语料 **>20,000 h / 480+ 场景 / 4.32M+ episode**；公开子集 **2,000 h**。重建成功率与 WBC 回放成功率均约 **98%**。
- **对 wiki 的映射：**
  - [人形训练数据管线](../../wiki/queries/humanoid-training-data-pipeline.md)
  - [双臂操作](../../wiki/tasks/bimanual-manipulation.md)

### 3) 基线与 zero-robot 后训练（§5–6.2）

- **骨干：** StarVLA-QwenPI（VLA）、OpenPI-π₀.₅（VLA）、LingBot-VA（WAM）。
- **协议：** UMI-only post-training vs 同场景真机遥操作 post-training；直接真机部署。
- **结果：** 三骨干与 teleop 对齐到几个百分点内；场景偏移下仍具泛化；精密插入最强 **85%**。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [StarVLA](../../wiki/methods/star-vla.md)

### 4) 大规模 UMI 预训练（§6.3）

- **4000 h** 同语料预训练：10 未见任务动作误差 **-41%**；再后训练时 StarVLA-QwenPI 真机 **+18.1 pp**。
- **对 wiki 的映射：**
  - [具身数据金字塔](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)（若存在）或 [manipulation](../../wiki/tasks/manipulation.md)

### 5) 开源核查（步骤 2.5；初检 2026-07-30，复检 2026-08-04）

| 项 | 状态 |
|----|------|
| 项目页 | <https://cloud.simpleai.tech/simple-world-lab/hifi-umi/> — 数据叙事与媒体；Resources 仅论文 + HF |
| Hugging Face | **已公开** `simple-world-lab/HiFi-UMI-2K`（CC BY 4.0，LeRobot v3 风格；约 35k downloads） |
| 采数硬件/训练代码 | 项目页 JS 资源与 GitHub 检索 **仍未列** 完整硬件 BOM 或训练仓库 URL → **部分开源：数据已开，系统代码未列** |
| 结论 | 数据集可复用；「zero-robot」实验复现依赖自备骨干与部署栈 |

### 6) 数据接口要点（HF card，训练消费）

- **state/action：** 各 20 维 = 右 10d + 左 10d；每手 `[xyz(m), rot6d(6), gripper(rad)]`；`action` 为 **绝对 next-state 目标**（论文训练时可再转相对动作）。
- **有效帧：** 保留 `valid.frame == false` 以对齐视频时间戳；训练应过滤为 `true`。
- **六视角 key：** `head_main` / `head_main_stereo_right` / `left|right_hand_up|down`。
- **评测规模提示：** VLA 对照约 **3200** 条 HiFi-UMI vs **300** 条 teleop / 任务（管线对比，非等样本效率）。
- **对 wiki 的映射：**
  - [HiFi-UMI 论文实体](../../wiki/entities/paper-hifi-umi.md)
  - [具身数据金字塔](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)

## 关键数字速查

| 指标 | 数值 |
|------|------|
| 公开子集 | **2,000 h**（HiFi-UMI-2K） |
| 源语料 | **>20,000 h**，480+ 场景，4.32M+ ep |
| EE 局部精度 | **~3 mm** |
| 跨传感器同步 | **<40 µs** |
| 相机视角 | **6** / episode |
| vs teleop ΔSR | **-2.5 / +3.1 / -0.6 pp**（三骨干） |
| 精密插入 | **85%**（最强） |
| 4k h 预训练 | 未见任务误差 **-41%**；真机再 **+18.1 pp** |

## 其他公开资料

- **项目页：** [sites/hifi-umi-project.md](../sites/hifi-umi-project.md)
- **数据集：** [datasets/hifi-umi-2k.md](../datasets/hifi-umi-2k.md)
- **arXiv HTML：** <https://arxiv.org/html/2607.25895>

## 当前提炼状态

- [x] sources 归档
- [x] 升格 wiki 实体页
- [x] 交叉 HandUMI / teleop / VLA / WAM
