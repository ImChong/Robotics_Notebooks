# Human-as-Humanoid: Enabling Zero-Shot Humanoid Learning from Ego-Exo Human Videos with Human-Aligned Embodiments

> 来源归档（ingest）

- **标题：** Human-as-Humanoid: Enabling Zero-Shot Humanoid Learning from Ego-Exo Human Videos with Human-Aligned Embodiments
- **类型：** paper / project page
- **原始链接：** <https://zgc-embodyai.github.io/Human-as-Humanoid/>
- **机构：** ZGC EmbodyAI 等
- **入库日期：** 2026-07-03
- **更新日期：** 2026-07-22
- **开源状态：** 未开源；截至 2026-07-22，项目页未列官方 GitHub、代码仓库、数据下载或可运行 README，仅提供项目页视频、摘要和 BibTeX。
- **一句话说明：** 以 **PrimeU** 人形对齐本体为底座，将同步 **Ego-Exo** 人类视频经运动恢复与分阶段 IK 转为 **60-DoF 控制器对齐动作块**，训练 **PhysDex**（DS-HKC FK 感知监督）；转换链约 **20 FPS**，示范吞吐较遥操作高 **4.8–7.2×**，可在无目标任务机器人示范下部署真实高 DoF 操作策略。

## 摘要要点

- VLA 需要高质量 observation-action supervision，但高 DoF 人形遥操作昂贵、效率低；人类 ego 视频丰富却缺少可执行机器人动作标签。
- Human-as-Humanoid 联合对齐 **机器人形态、传感视角和 action-label 接口**，让人类视频可用于高 DoF VLA 训练。
- **PrimeU** 提供人类比例上身形态：双 7-DoF 手臂、双 20-DoF 灵巧手、3-DoF 颈、3-DoF 腰，总计 60 DoF；头部与腕部 RealSense D435 对齐部署视角。
- 同步 ego-exo 视频中，ego 用于策略输入，exo 用于遮挡更少的上肢/手部 motion recovery。
- PhysDex 通过 **DS-HKC** 可微 FK 保持腕和指尖任务空间几何，而不只是逐维拟合关节数值。

## 方法要点

- **Embodiment alignment：** PrimeU 的肩宽、手长、臂展与成人男性操作尺度接近，降低重定向前的形态 gap。
- **Observation-motion compatibility：** Ego 流提供 deployment-aligned observation，Exo 流支持更稳健的人体/手部恢复。
- **Action-interface alignment：** 输出动作遵守 PrimeU URDF、关节顺序、关节限制和控制器接口，避免只得到任务空间意图。
- **Joint-task consistency：** DS-HKC 将 60 维关节动作通过 FK 投影到腕/指尖几何，约束接触相关任务空间结构。
- **转换管线：** ego-exo 视频 → tracking / mesh-aware motion recovery → staged IK → controller-aligned 60-DoF action chunks → PhysDex VLA post-training。

## 实验与数字

- **转换吞吐：** 项目页报告人类视频到动作标签转换约 **20 FPS**。
- **采集效率：** 相对 teleoperation 的 raw demonstration throughput gain 为 **4.8–7.2×**。
- **动作空间诊断：** human-only action tokenizer 在 100 个真实机器人 evaluation windows 上 cross-domain mean normalized MAE 为 **0.0080**，mean EE error 为 **5.34 mm**。
- **zero-shot 任务：** magic-cube packing、cup stacking、ring toss、water pouring 使用 human-only converted data，无目标任务机器人示范。
- **few-shot 任务：** light-bulb installation、temperature sensing 使用 human converted labels 加少量真实机器人数据。

## 开源 / 复现状态

- **代码：** 未发现官方代码仓库。
- **数据：** 未发现公开数据下载链接。
- **项目页核查：** 2026-07-22 检查 <https://zgc-embodyai.github.io/Human-as-Humanoid/>；页面含 demo、摘要、PrimeU 说明、诊断表和 BibTeX，但无 GitHub / Code / Dataset 按钮。
- **复现边界：** 外部复现需要 PrimeU 或等价人形上身、URDF/控制器接口、ego-exo 标定与运动恢复/IK 实现；目前无法从官方公开材料直接运行。

## 对 wiki 的映射

- [paper-human-as-humanoid](../../wiki/entities/paper-human-as-humanoid.md) — 完整论文实体页，含流程图、机制、工程状态与局限。

## 参考来源（原始）

- 项目页：<https://zgc-embodyai.github.io/Human-as-Humanoid/>
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
