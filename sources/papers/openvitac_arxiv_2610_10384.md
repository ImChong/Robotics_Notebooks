# OpenViTac 论文来源归档

## 元数据

- **题目：** OpenViTac: Learning and Benchmarking Visuo-Tactile Policies in a Unified Sim-and-Real Framework
- **方法：** OpenViTac 基准 + OpenVTLA（π0.5 触觉增强）
- **作者：** Yifan Wu, Qin Li, Nan Min, Guojin Zhong, Haoyu Zhao, Zhiyuan Li, Houze Xu, Shengqi Xu, Xingyao Lin, Zijie Diao, Zhaoxiang Liu, Shiguo Lian, Shunlin Lu, Shihao Zhao, Ziyi Ye, Zuxuan Wu, Yu-Gang Jiang（共 17 人）
- **单位：** 据项目页：复旦大学、上海市多模态具身智能重点实验室、合肥工业大学、新加坡国立大学、新智具身智能（NeoteAI）、中国联通。
- **预印本：** https://arxiv.org/abs/2610.10384
- **HTML：** https://arxiv.org/html/2610.10384v1
- **版本：** v1，2026-10-07，cs.RO，CC BY 4.0
- **项目页：** https://fvl-repo.github.io/OpenViTac/ （归档见 sources/sites/openvitac-project.md）
- **代码与数据：** 项目页与论文均未给出 GitHub/Hugging Face/数据集链接，也无明确发布承诺；截至 2026-10-12 未核实到公开代码。

## 论文要解决的问题

视觉-触觉-语言-动作（VTLA）策略进展很快，但缺少能同时在仿真与真机上评测触觉操作的统一基准；不同研究的触觉传感器、仿真环境、机器人接口不一致，复现与 Sim-to-Real 评测困难。

## 方法与训练信息

1. **基准：** 11 个仿真任务与 8 个配对真机任务（8 个共享任务用于仿真-真机匹配比较），按四个依赖触觉的能力维度分组：属性感知（重量/硬度/粗糙度分类、粗糙度引导重抓、空罐选择）、易碎感知（抓薯片）、接触丰富（齿轮装配、拉抽屉）、精度（插 USB、插块 v1/v2）。仿真基于 Isaac Sim 5.1.0 + Isaac Lab 2.3.0，触觉仿真基于 TacEx；覆盖 GelSight、Xense、NeoteAI 三类传感器与 Franka Hand、Robotiq 两种夹爪（机械臂型号文中未指明）。
2. **被评策略：** VLA（π0.5、StarVLA-Qwen3、InternVLA-A1.5、XVLA、Lingbot-VLA-2、Xiaomi-Robotics）；WAM（FastWAM、Lingbot-VA、StarVLA-Cosmos）；VTLA（FTP-1 与提出的 OpenVTLA）。
3. **触觉表征与融合研究：** 表征比较 SigLIP、T3、Sparsh、AnyTouch2；融合策略比较 token 级拼接、特征级调制（FiLM）、动作级条件、触觉驱动专家、触觉引导精修五种。
4. **OpenVTLA：** 以 π0.5 为骨干，用 AnyTouch2 编码触觉历史为紧凑 token，投影到 VLM 隐层维度后前置到视觉 token 序列（token 级拼接），不新增融合模块或触觉专家；flow-matching 行为克隆，20k 步，全局 batch 64。
5. **数据：** 仿真每种传感器配置 1,300 条回合、11 任务、约 918K 帧；真机每种传感器配置 2,400 条示范、8 任务、约 2.1M 帧。每任务训练一个策略，通常 100 条回合（In-B v2 为 300）。

## 论文报告的结果

- 仿真平均成功率：OpenVTLA 68.7%；FTP-1 61.2%；InternVLA-A1.5 51.5%（最佳 VLA）；π0.5 48.6%；Xiaomi-Robotics 40.0%；FastWAM 37.4%（表中最佳 WAM）；Lingbot-VA 29.4%。
- 真机平均：OpenVTLA 54.6%；FTP-1 51.9%；InternVLA-A1.5 43.1%；π0.5 41.0%；FastWAM 29.0%。共享任务上 OpenVTLA 比 π0.5 高 13.6 点（仿真）/ 13.5 点（真机）。
- 仿真-真机一致性：七种方法在八个共享任务上 Pearson r = 0.967、MMRV = 0.006（论文正文数字；项目页写 0.913）。
- 表征（同拼接接口）：AnyTouch2 68.7%、Sparsh 64.7%、SigLIP 59.5%、T3 58.1%。融合：拼接 68.7%、专家（T3）61.2%、调制（Sparsh）56.5%、精修 50.4%、动作级 48.6%。
- 消融：AnyTouch2 随机初始化降至 56.5%；用重复当前帧代替触觉历史降至 58.4%。
- Sim-Real 联合训练（每任务 100 条真机示范 + 0/250/500 条仿真示范，每设置 20 次试验）：In-USB 上 π0.5 15%/25%/40%、OpenVTLA 20%/30%/40%；齿轮装配上 π0.5 35%/45%/50%、OpenVTLA 50%/50%/60%；500 条仿真示范平均增益 π0.5 约 20 点、OpenVTLA 约 15 点；光照扰动下 OpenVTLA 500 仿真示范达 30%（In-USB）/60%（GA）。

## 复现与解释边界

- 论文自陈触觉观测仍有 Sim-Real 差距（触觉渲染、传感器建模、标定）；后续方向含可变形物体、表面交互、长程家务与灵巧手。
- 原文内部有不一致：引言称最强 WAM 基线仿真 53.3%/真机 48.5%，而表 2/3 中最佳 WAM 仅 37.4%/29.0%；项目页与正文的相关系数也不同。引用具体数字前需对照原文。
- 每任务一个策略、每设置 20 次试验，统计功效有限；未发布代码或数据，基准无法直接复用。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-openvitac.md
- 概念入口：wiki/concepts/visuo-tactile-fusion.md、wiki/concepts/tactile-sensing.md
- 周报汇总：wiki/overview/frontier-manipulation-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10384>
- arXiv HTML：<https://arxiv.org/html/2610.10384v1>
- 项目页：<https://fvl-repo.github.io/OpenViTac/>
