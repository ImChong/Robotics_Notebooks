---
type: entity
project_id: ego2robot-synthesis
project: https://www-ye.github.io/ego2robot_blog/
code: https://github.com/www-Ye/ego2robot_blog
tags: [paper, robot-data, egocentric, vla, synthesis, ruc, alibaba, shanghaitech, bigai, buaa]
status: complete
updated: 2026-11-14
arxiv: "2608.02580"
related:
  - ../methods/vla.md
  - ../methods/egoscale.md
  - ../methods/wilor.md
  - ./robotwin.md
  - ./paper-egoverse.md
  - ./paper-embodiedvae.md
  - ./paper-gsr-paravla.md
  - ./paper-ego2robot-learning-survey.md
sources:
  - ../../sources/papers/ego2robot_arxiv_2608_02580.md
  - ../../sources/sites/ego2robot-blog.md
  - ../../sources/repos/ego2robot-ye-code.md
  - ../../sources/blogs/wechat_embodied_station_ego2robot_mango_grasp_2026-08-11.md
original:
  - ../../sources/papers/ego2robot_arxiv_2608_02580.md
summary: "Ego2Robot（arXiv:2608.02580，人大/阿里Qwen/上科大/BIGAI）：第一人称人视频经重定向、机器人视觉合成和质检得到 15 形态 18,561 h 数据，和机器人示范共同预训练改进解耦 OOD；合成管线与 RoboTwin 评测代码现已公开。"
---

# Ego2Robot：第一人称人视频规模化合成机器人数据

**Ego2Robot**（*Scalable Robot Data Synthesis from Egocentric Human Data*；[arXiv:2608.02580](https://arxiv.org/abs/2608.02580)，[项目页](https://www-ye.github.io/ego2robot_blog/)）由 **中国人民大学 AIM3 / 阿里通义千问 / 上海科技大学 / 北京通用人工智能研究院** 等提出：把大规模第一人称操作视频对齐成 **可给 VLA 预训练的机器人格式数据**。

## 一句话定义

**用动作重定向、去手后渲染机械臂、再做三级质检，把人的第一人称操作变成 15 种本体上的相机系相对 EEF 轨迹——用来补机器人遥操作覆盖不到的场景与任务多样性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EEF | End-Effector | 末端；本文输出相机系相对位移 |
| VLA | Vision-Language-Action | 下游预训练对象 |
| IK | Inverse Kinematics | 为每条轨迹搜可解的基座位姿 |
| OOD | Out-of-Distribution | 解耦扰动下的泛化 |
| TCP | Tool Center Point | 由拇指与虚拟指尖中点定义 |

## 为什么重要

- Open X / DROID / AgibotWorld 仍受硬件与遥操作成本限制；人视频天然场景多、任务杂。
- 小规模 retarget-and-render 已证明能训单任务策略；本文问的是 **能否当 VLA 预训练资产**。
- 把 RoboTwin 2.0 的捆绑 OOD 拆成视觉 / 布局 / 本体 / 语义，才能看清合成数据到底补哪一轴。

## 核心信息

| 项 | 内容 |
|----|------|
| **规模** | 约 1,940 h 人视频 → **18,561 h** 合成机器人数据；共训机器人数据约 6,565 h（含 InternData 仿真） |
| **形态** | 15 种：Panda、UR5e、ARX-L5、xArm7、Aloha-Agilex 等 |
| **源** | ANT / EgoDex / ViTRA / EgoVerse |
| **开源** | **代码已开放（2026-11-12 重新核查）**：`www-Ye/ego2robot_blog` 含合成管线、LeRobot 转换与 RoboTwin 评测代码；原“仅静态站”是 2026-08-15 历史快照。完整 18,561 h 合成数据、权重与第三方依赖不能假设直接提供。 |

## 核心原理

### 方法栈

Path A 吃已标注手姿态；Path B 用 [WiLoR](../methods/wilor.md) + DynHaMR。共享三段：手关键点→夹爪 TCP/开合/姿态并平滑；SAM 3 + ProPainter 去臂后做基座网格搜索与深度合成；L1/L2/L3 质检。动作用 **相机系相对 EEF**，避开未知外参。训练时按源降采样，把人手速度拉近遥操作。

### 流程总览

```mermaid
flowchart TB
  ego["Ego 视频 / 标注"]
  hand["手姿态 Path A/B"]
  act["动作对齐 → 相机系 EEF"]
  vis["去手 + IK 基座 + 深度合成"]
  qc["L1/L2/L3 质检"]
  vla["与真机数据共训 VLA"]
  ego --> hand --> act --> vis --> qc --> vla
```

## 源码运行时序图（2026-11-12 官方 main 静态核对）

源码入口来自 [作者的 `scripts/run_pipeline.py`](https://github.com/www-Ye/ego2robot_blog/blob/main/scripts/run_pipeline.py)，阶段编排核对 [`ego2robot/pipeline.py`](https://github.com/www-Ye/ego2robot_blog/blob/main/ego2robot/pipeline.py) 与 [`configs/egoverse_mecka.yaml`](https://github.com/www-Ye/ego2robot_blog/blob/main/configs/egoverse_mecka.yaml)。下图对应**EgoVerse 已有手姿态标注的默认配置**，不是假定所有数据都统一经过 raw-video 的手重建。

```mermaid
sequenceDiagram
  autonumber
  actor U as 用户
  participant CLI as scripts/run_pipeline.py
  participant P as ego2robot/pipeline.py
  participant W as joblib worker
  participant B as base_egoverse processor
  participant A as action / smoothing
  participant V as arm_segmentation / hand_inpaint
  participant R as robot_inpaint
  participant FS as episode files
  U->>CLI: python scripts/run_pipeline.py --config configs/egoverse_mecka.yaml
  CLI->>CLI: OmegaConf.load + CLI overrides
  CLI->>P: run_parallel(cfg)
  P->>P: resolve_modes + get_demo_folders
  loop 按 config.mode 依序执行各阶段
    P->>P: check_exists + split demos
    P->>W: joblib Parallel(_process_chunk)
    W->>W: load_processor_class(stage)
    alt base_egoverse
      W->>B: process_one_demo(demo)
      B->>FS: 保存基础输入 / 中间结果
    else action 或 smoothing
      W->>A: process_one_demo(demo)
      A->>FS: 保存动作 / 平滑中间结果
    else arm_segmentation 或 hand_inpaint
      W->>V: process_one_demo(demo)
      V->>FS: 保存分割 / 修补结果
    else robot_inpaint
      W->>R: process_one_demo(demo)
      R->>FS: 保存 NPZ / MP4 合成输出
    end
    W-->>P: stage result + JSONL log
  end
  P-->>CLI: pipeline finished
  opt 独立导出命令，不属于 run_parallel
    U->>FS: scripts/convert_to_lerobot.py 读取 NPZ/MP4
    FS-->>U: 转换为 LeRobot 数据集
  end
```

源码中的 `STAGE_DEFS` 维护阶段到 Processor 的映射；`run_parallel` 外层按配置逐 stage 运行，`_process_chunk` 为每个 demo 调用 `process_one_demo`。README 的**原始视频**模式另需 `subtask → depth_pose → hand → ...`，不能将它和已标注视频模式混为一谈。LeRobot 导出为独立脚本，未运行完整生成任务。具体文件及依赖见 [源码一手归档](../../sources/repos/ego2robot-ye-code.md)。

## 工程实践

| 项 | 建议 |
|----|------|
| 源码运行时序图 | **已按当前官方源码补充**；本地未执行 GPU 和训练复现 |
| 混合比 | 项目页默认 **1:1** 合成:真机；本体对齐可试 3:1 |
| 动作空间 | 跨相机、跨形态时优先相机系相对 EEF，不要强行世界系 |
| 真机补数据 | 现场拍几分钟 ego-play，再走同一管线混进 few-shot 微调 |
| Franka | 大运动学缺口上零样本仍弱（<7%），不要指望合成数据单独打穿 |

## 实验与评测

固定 Qwen3.5-4B + DiT、同等 19.2M 帧预算。Ego2R+Robot **1:1**：Clean 62.2→**68.1**，Visual **67.3**，Task **54.1**。**3:1** 在 Embody（28.2）与 EBench（51.7，+12.1）更好。未见物体 29.3→40.0。仅 ego 预训练：生视频 28.1 → 15 形态管线 33.5 → 再加生视频 37.3。ARX ACone 五任务、每任务 20 条遥操作：Mix + Ego2R Play 全最高。

## 与其他工作对比

相对 [EgoScale](../methods/egoscale.md)：EgoScale 在人视频上预训练再 mid-train 对齐；Ego2Robot 先把人视频 **渲染成机器人像素+动作** 再共训。相对 RoviAug / Mirage：那些补的是机–机外观，这里跨的是人–机。相对 [RoboEdit](./paper-roboedit.md)：RoboEdit 输出 **full robot interaction video + 3D hand states**（RoboEdit-14M），Ego2Robot 输出 **相机系相对 EEF 轨迹** 供 VLA 预训练。相对 [RoboTwin](./robotwin.md)：本文扩展其评测轴，不是替代数据生成器。

## 结论

**人视频要变成 VLA 预训练资产，关键是对齐动作、外观和速度，而不是把原始 ego 像素直接扔进共训。**

1. **1:1 共训是默认甜点** — Clean 与多数 OOD 轴一起涨。
2. **视觉与语义收益最大** — 场景多样 + 多形态上色解释 background/lighting/color 与未见物体。
3. **本体迁移部分有效** — ARX/UR5 有增益，Franka 仍近乎失败。
4. **生视频不是废物** — 管线后仍可当第 16 种「形态」混入。
5. **真机 few-shot 用现场 ego-play** — 比只靠离线合成更贴部署场景。
6. **合成管线已开源** — 官方仓提供运行入口与 RoboTwin 评测；重跑仍需原始视频、DA3/SAM 3/ProPainter 等第三方模型与 GPU，未核实作者 18,561 h 全量产物可下载。

## 局限与风险

- 夹爪假设（平行爪 TCP）覆盖不了多指灵巧接触。
- 质检 L3 依赖 VLM，可能系统性漏过细交互错误。
- 合成臂外观与真实灯光/材质仍有域差；Franka 数字说明形态鸿沟不会被渲染抹平。
- 当前可读代码不是完整开箱即用的论文复现实验：需要额外的模型权重、数据路径、依赖和本体/相机标定；仓库静态核对不能等同于跑通训练或验证论文结果。

## 关联页面

- [Ego2Robot Learning 综述（arXiv:2611.13610）](./paper-ego2robot-learning-survey.md) — 归纳 EHV→机器人操作的四范式与数据/评测要求，**不同于本页的单项数据合成算法**。

- [VLA](../methods/vla.md)
- [EgoScale](../methods/egoscale.md) — 人视频规模化的另一条对齐路径
- [KODA](./paper-koda-egocentric-vla.md) — 跨控制器对齐人类第一视角示范的人形 VLA 数据
- [WiLoR](../methods/wilor.md) — Path B 手估计
- [RoboTwin 2.0](./robotwin.md) — 解耦评测宿主
- [EgoVerse](./paper-egoverse.md) — 954 h 源之一
- [GSR / ParaVLA](./paper-gsr-paravla.md) — 指令改写轴与本文 Task/Lang 扰动互补
- [EmbodiedVAE](./paper-embodiedvae.md) — 表征侧紧凑可控，对照数据侧规模化
- [RoboEdit](./paper-roboedit.md) — 人类视频编辑为 robot video + 3D hand states（RoboEdit-14M）

## 参考来源

- [Ego2Robot 论文摘录](../../sources/papers/ego2robot_arxiv_2608_02580.md)
- [Ego2Robot 项目页归档](../../sources/sites/ego2robot-blog.md)
- [官方开源代码、README 与实际时序调用归档](../../sources/repos/ego2robot-ye-code.md)
- [具身智能小站 9 篇盘点](../../sources/blogs/wechat_embodied_station_ego2robot_mango_grasp_2026-08-11.md)
- [arXiv:2608.02580](https://arxiv.org/abs/2608.02580)

## 推荐继续阅读

- [Ego2Robot 项目页](https://www-ye.github.io/ego2robot_blog/)
- [官方合成管线及 RoboTwin 评测代码](https://github.com/www-Ye/ego2robot_blog)
- Yuan et al., [Qwen-RobotManip](https://arxiv.org/abs/2606.17846) — 文内称使用本合成数据
