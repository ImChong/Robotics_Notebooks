# AnchorVLN: Geometry-Anchored Vision-Language Grounding Reasoning for Open-Vocabulary Navigation

- **作者：** Long Giang Vu*, Chengkai Yao*, Yuxin Liu, FNU Aryan, Rajath Chandrashekar Aralikatti（* 同等贡献）
- **机构：** 原文署名栏仅列个人邮箱（未显式标注机构名称；文中提及与 SysNav 来自同一实验室，测试基于 CMU VLN Challenge 2026 平台）
- **版本 / 载体：** arXiv:2609.12285v1 [cs.RO]（Submitted on 10 Sep 2026，arXiv HTML 全文）
- **原文：** https://arxiv.org/abs/2609.12285 （全文读取载体：https://arxiv.org/html/2609.12285 ）
- **项目页 / 代码：** 已开源 — 官方 MCP 服务与代理实现仓库 [aryanmangal769/embodied-nav-mcp](https://github.com/aryanmangal769/embodied-nav-mcp)（MIT 协议，主分支 `main`；另集成并调用 CMU VLN Challenge 2026 参赛基线仓库 [ginlov/CMU-VLN-Challenge-2026](https://github.com/ginlov/CMU-VLN-Challenge-2026) 的几何感知与 ROS 通信模块）
- **核对范围：** 论文全文全部章节（Abstract、Sec. I–VI、Table I–III、Fig. 1–4 图注、Appendix A 及 Table IV），以及官方仓库 `README.md` 与源码目录结构清单
- **通读：** 已完成

## 原文核对笔记

- **核心设计原则：** 「VLM 提出语义，几何决定度量」（the VLM proposes semantics; geometry decides metrics）。系统分为三层权威与两道边界：VLM 客户端仅能使用自然语言短语和不透明句柄（如 `obj_7`），`Embodied-Nav-MCP` 工具服务器维护持久三维物体图并将句柄映射为 LiDAR 测距与路点，底层未改动自主导航栈接收 `Pose2D` 请求并执行局部规划。
- **工具接口核对：** 论文 Table I 给出 10 个核心工具（感知：`ground`、`expand_vocab`、`verify_present`、`nearest`、`farthest`、`between`；运动：`drive_to`、`explore`、`orbit`；输出：`publish`）。GitHub 官方实现 `embodied-nav-mcp` 在此基础上还暴露了面向挑战赛全部三类题型的 8 工具运行接口（含 `get_view`、`stop` 及计数任务 `parse_numerical`、`count_in_view`、`commit_count`、`publish_object`）。
- **关键几何保护机制：**
  - `ground` 将 360° 全景图展开为球面切面（gnomonic faces），VLM 给出 2D 检测框后反投影为方位角，取窄锥内 LiDAR 点云的**中位数距离**作为深度（避免前景与背景均值混叠）。
  - 三道拒绝门控（elevation floor、implied-size gate、empty-cone test）拦截不可靠测距；受门控的句柄标记为 `unmeasured`，调用 `drive_to` 时沿方位角试探性前进一定步长以改变观测几何。
  - `drive_to` 在服务端复现底层控制器对地形图的目标函数，预测真实停靠点并发送修正路点（Platform Model），同时通过里程计直接判断到达。
- **实验核对（CMU VLN Challenge 2026 开发集，15 场景）：**
  - 指令跟随（30 题）：完整系统 Challenge Score **64.4%**（nDTW 0.360，TL ratio 1.23，最近接近距离 1.58 m）。移除控制器模型（w/o Platform Model）分数骤降 13.3 个百分点至 **51.1%**（$t = 2.77$）；移除 LiDAR 测距提升（w/o Lift）为 **63.6%**（$-0.8$ pp，$t = 0.23$，受底层地形图重规划掩盖，但测距提升每题可减少 1.45 次 grounding 调用，$t = 2.86$）；两者均移除（Naive）为 **54.4%**（$-10.0$ pp，$t = 2.38$）；移除提示词中的访问历史记忆（w/o Memory）为 **68.3%**（$+3.9$ pp，$t = 1.13$，低于单配置重复运行波动 5.8 pp 及最小可分辨差 9.4 pp）。
  - 物体指代（45 题，3D IoU $\ge 0.1$ 命中阈值）：完整系统命中率 **22%**（10/45，Mean IoU 0.060，中位中心误差 2.48 m，命中时均值 36 cm）；移除 MCP 结构（naive）降至 **18%**（Mean IoU 0.054，场景图均值由 52.4 个物体萎缩至 43.8 个）；移除几何落地由模型直接输出坐标（model-metric）命中率为 **0%**（0/45，Mean IoU 0.000，中位中心误差 3.37 m）。
- **对 wiki 的映射：** [AnchorVLN：几何锚定视觉-语言推理导航系统](../../wiki/entities/paper-agentic-2609-12285.md)
