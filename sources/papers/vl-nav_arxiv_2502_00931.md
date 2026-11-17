# VL-Nav: Neuro-Symbolic Reasoning-based Vision-Language Navigation（一手论文阅读归档）

- **类型：** paper / 机器人研究获奖论文
- **标题：** VL-Nav: Neuro-Symbolic Reasoning-based Vision-Language Navigation
- **中文题名：** VL-Nav：神经符号推理驱动的视觉语言导航
- **arXiv：** <https://arxiv.org/abs/2502.00931>
- **原文 HTML：** <https://arxiv.org/html/2502.00931>
- **PDF：** <https://arxiv.org/pdf/2502.00931>
- **版本：** v7（2026-07-14）
- **会议奖项线索：** IROS 2026 Best Paper Award on Cognitive Robotics（奖项由二手综述提出，与一手项目资源交叉核对）
- **项目页：** <https://sairlab.org/vlnav/>
- **核对范围：** arXiv HTML v7：摘要、§I–V、III-A 任务规划与记忆、III-B frontiers/IBTP/NeSy 评分、III-C 路径规划、DARPA TIAMAT 仿真和实机报告；官网资源区代码发布状态。
- **通读：** 已完成
- **开源核验：** 项目页提供视频与论文，但未发现 VL-Nav 系统本体官方可运行仓库；所引用 YOLO-World/FastSAM 等依赖开源不等于本系统开源
- **入库日期：** 2026-11-17

## 要点摘录与 wiki 映射

1. 系统分成 NeSy Task Planner、NeSy Exploration、底层 Goal Selection & Path Planning。
2. 目标评分综合视觉语言相似度、未知区域收益、距离，而非纯 VLM 给出目的地。
3. 实验含多层楼和远端 RTX 4090，不能视为纯板端离线 VLM 实验。

- **技术机制：** NeSy Task Planner 将命令分为 explore / go-to 原子任务，使用房间/物体组成的 3D 场景图与物体最佳视角照片作联合记忆并触发重新规划；NeSy Exploration 从 frontier 与 IBTP 生成候选目标，根据视觉语言语义、未知区域收益和路程距离打分；底层局部导航在选定目标上执行。
- **评测证据：** 论文报告 DARPA TIAMAT 仿真 **83.4% 室内**、**75.0% 室外**任务成功率；真实环境 **86.3%** 成功率，展示最长 **483m** 导航任务及多层楼演示。指标仅适用于论文测试场景。
- **局限：** 依赖可靠的开放词汇检测、场景图构建、位姿估计和模型推理；实机系统还需设备/云端计算协同。官方未发布完整系统代码，暂无法按论文完全复现。
- **对 wiki 的映射：** [VL-Nav：神经符号推理驱动的视觉语言导航](../../wiki/entities/paper-vl-nav.md)（每篇论文只创建一个独立详情节点）
- **二手索引：** [2026 四大顶会 12 篇获奖论文](../blogs/wechat_shenlan_four_robotics_conferences_12_awards_2026_11_17.md)（仅奖项线索，非技术证据）
