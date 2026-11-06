# Awesome World Models README 通读（目录结构与关键挑战）

> 来源归档（原文通读 · 支撑 `wiki/overview/sun-awesome-wm-technology-map.md`）

- **标题：** Awesome World Models（sun254667）README.md
- **类型：** repo / awesome-list / curated-index
- **原文：** <https://github.com/sun254667/awesome-world-models>（2026-11 拉取 main 分支 README.md 全文，约 3600 行）
- **一句话说明：** 按「综述 → 教程 → 奠基论文 → 核心范式 → 交互式 → WAM/VLA → 基于模型的 RL → 应用域 → 数据集 → 训练与推理技术 → 仿真平台 → 理论」组织的世界模型清单，末尾给出关键研究挑战。
- **核对范围：** 2026-11 复读 main 分支 README.md 全文（3622 行）：目录、收录标准、§1–§3（综述 / 教程 / 奠基论文）、§4–§12 各小节的全部条目（标题 + 亮点；§8.1.5 之后的长列表仅核对标题）、§13 相关清单、§14 关键研究挑战、许可。数据集表（§9）核对了名称 / 年份 / 任务列。未打开清单所链接的各篇论文原文，条目的技术声明均为清单作者的一句话亮点，未经独立核实。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **收录标准：** 优先 CVPR / ICCV / ECCV / NeurIPS / ICLR / ICML / RSS / AAAI / TPAMI 等顶会顶刊；覆盖语言 / 视觉视频 / 3D-4D / 潜空间 / 物体中心 / 符号与神经符号 / 交互式 / WAM / VLA / 基于模型 RL / 机器人 / 自动驾驶 / GUI 智能体 / 科学仿真。
- **顶层结构：** 16 个顶层节；核心范式 4.1–4.7（语言、视觉视频、3D/4D、潜空间、物体中心、符号、通用基础）；交互式 5.1–5.2；WAM 与 VLA 6.1–6.3（级联、联合统一、训练与评测）；基于模型 RL 7.1–7.3（经典系列、连续控制、离线与泛化）；应用域 8.1–8.5（机器人含基础 / 操作 / 导航 / 足式 / 与策略学习 / 策略评测六个子类，另有自动驾驶、GUI、科学发现、社会经济仿真）。
- **后半结构：** 数据集与基准（9.1–9.3）、训练与推理技术（记忆机制、后训练与测试时扩展、高效实时推理）、仿真平台与物理引擎（11.1–11.7）、理论与立场（12.1–12.2）。
- **奠基论文节：** 列出 OpenAI Sora 技术报告（视频生成作为世界模拟器）、LeCun《A Path Towards Autonomous Machine Intelligence》、Ha & Schmidhuber《World Models》（称其为把世界模型引入深度学习的开创性工作）。
- **相关清单：** 自动驾驶、机器人、医疗、General World Models Survey、视频生成中的物理认知等并列清单。
- **关键研究挑战（原文列出）：** 持久状态核心、物理落地（可执行的仿真器）、指令泄漏、反事实可控性、模型失配下的重规划、反应能力评估（对偏离日志的行为是否响应）、模态感知蒸馏、可验证性。
- **许可：** CC0 1.0。

## 对 wiki 的映射

- [`wiki/overview/sun-awesome-wm-technology-map.md`](../../wiki/overview/sun-awesome-wm-technology-map.md)
- [`wiki/entities/awesome-world-models.md`](../../wiki/entities/awesome-world-models.md)

## 参考来源

- 仓库：<https://github.com/sun254667/awesome-world-models>
