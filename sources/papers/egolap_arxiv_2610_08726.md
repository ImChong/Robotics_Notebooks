# EgoLAP 论文来源归档

## 元数据

- **题目：** EgoLAP: Learning from Egocentric Human Data through Language-Action Reasoning
- **方法：** EgoLAP（Egocentric Language-Action Pre-training，名称展开以论文为准，摘要页未展开）
- **作者：** Lihan Zha, Shresth Grover, Tenny Yin, Samuel M. Bateman, Hengkai Pan, Mengchao Zhang, Aykut Onol, Allen Z. Ren, Dhruv Shah, Anirudha Majumdar
- **单位：** arXiv 页面未列出机构（周报写作普林斯顿大学等；作者含 Majumdar、Shah 等，机构未在 arXiv 页面核实）
- **预印本：** https://arxiv.org/abs/2610.08726
- **HTML：** https://arxiv.org/html/2610.08726v1
- **版本：** v1，2026-10-06，cs.RO（交叉 cs.AI），CC BY 4.0
- **项目页：** https://ego-lap.github.io/（2026-10-12 打开核查：代码与检查点均标 “Coming soon”）
- **代码：** 截至 2026-10-12 未发布

## 论文要解决的问题

第一视角人类视频能扩大机器人学习数据，但人与机器人的身体结构不同，原始人体运动不是好的训练目标。论文主张两者不必共享低层动作，只需共享**运动意图**：把人手/末端的一秒运动写成结构化语言动作，并配以关于场景几何、物理和可供性的推理。

## 方法与训练信息

- **语言动作：** 每个样本概括一秒的手/末端执行器在自我中心坐标系下的运动；平移取整数厘米、旋转取 5° 倍数，舍入为零的分量省略；形如“move forward 5 cm, …, close gripper”，双臂用 “Left arm:” / “Right arm:” 子句。标签由位姿确定性生成，不需人工或 VLM 标注，也不需学习型 tokenizer。
- **人类数据：** 输入为头戴相机系下的一或两只手的跟踪位姿（可见正文未指明手部跟踪方法）；人与机器人运动规范化到以相机为中心的坐标系。
- **运动级推理标签：** 由 Gemini 离线生成（温度 0；MECKA 用 gemini-3.7-flash，DROID 用 gemini-3.5-flash），输入为指令、动作锚点和目标语言动作。人类推理来自 EgoVerse 的 MECKA 子集；EgoVerse 的 Scale、Aria 子集只提供语言与连续动作。
- **模型：** Mixture-of-Transformers；PaliGemma-2B VLM 主干 + 从零训练的约 300M 流匹配动作专家；输入最多三路 RGB（一路基座、两路腕部）、指令、离散成 256 档的状态；知识隔离（action expert 梯度不回传 VLM，VLM 只学文本目标）；动作视野 K=30，λ_text=0.8；推理 10 步 Euler、不生成文本。
- **数据混合：** 带推理标注的 DROID 与 MECKA 约占每个 batch 的 32%；ABC YAM、其他双臂数据集与其余 OXE 仅提供动作。总数据规模在可见正文中未给出（附录被截断，未核实）。
- **RL 后训练：** 在 ABC 仿真任务上对语言动作做 GRPO，冻结视觉与动作专家，奖励基于解析后的语言动作误差。

## 论文报告的结果

- **真机：** 定制双臂 YAM，零样本（臂间距与相机配置与所有训练本体不同）；五任务：Pick Place (Simple)、Fold Pants、Handover & Place、Wipe Object、Pick Place (Complex)；每任务 10 次（Complex 12 次），报告平均任务进度 ± SEM。EgoLAP 80.1%；无推理 44.5%；EgoFAST 35.2%；EgoOAT 18.0%。约 2.3× 来自 80.1% 对 35.2%。
- **仿真（14 任务）：** EgoLAP 38.3%；无推理 16.4%；EgoFAST 1.6%；EgoOAT 2.7%；比 ABC-VLA 高 16.9 个点。仅机器人 → 机器人+人类（仿真、无推理）：7.2% → 16.4%。
- **推理作用：** 运动级推理使 EgoLAP 真机 +35.6 点、仿真 +21.9 点；对 EgoFAST 加推理反而使真机 35.2% → 26.1%。
- **运动级 vs 组合式推理（仿真）：** 运动级推理相对纯语言动作 +21.9、相对 ECoT 式组合（子任务+物体框+视觉轨迹）+9.1、相对 FAST +7.1；组合式单独仅 5.3%，低于 16.4% 的无推理基线；先接组合式再接运动级为 14.4%，对比单用运动级 38.3%。
- **跨本体检索：** 推理使机器人→人 R@1 由 51% 升到 76%，人→机器人由 42% 升到 63%。
- **GRPO：** 仅采样语言动作而不生成推理 +19.4%；RL 中生成推理 +12.7%（推理偏离分布）。
- **文内不一致：** 引言中的“44.9 点”“44.7 点”与结果段的 35.6 点 / 9.3 点不一致，本归档采用结果段数字。

## 复现与解释边界

- 固定一秒窗口与量化运动可能漏掉快速、接触丰富的动态；评测仅限双臂桌面任务；RL 仅在仿真、优化的是运动匹配代理而非任务奖励（作者所述）。
- 推理标签依赖 Gemini 模型；真机任务数少、每任务 10 次。
- 代码与检查点“Coming soon”；数据集名称列出但未给下载链接。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-egolap-egocentric-language-action-reasoning.md
- 项目页归档：sources/sites/ego_lap_github_io.md
- 任务入口：wiki/tasks/bimanual-manipulation.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.08726>
- arXiv HTML 全文：<https://arxiv.org/html/2610.08726v1>（读取到前约 100,000 字符）
- 项目页：<https://ego-lap.github.io/>
