# SAGE: Scalable Agentic 3D Scene Generation for Embodied AI

- **作者：** Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, Shuran Song, Fangyin Wei
- **机构：** NVIDIA、伊利诺伊大学厄巴纳-香槟分校（UIUC）、康奈尔大学（Cornell University）、斯坦福大学（Stanford University）
- **版本 / 载体：** arXiv:2602.10116v2 [cs.CV]（v1 提交于 2026-02-10，v2 修订于 2026-02-20）
- **原文：** [arXiv:2602.10116 (HTML v2)](https://arxiv.org/html/2602.10116) / [arXiv 摘要页](https://arxiv.org/abs/2602.10116) / [项目页（重定向入口）](https://research.nvidia.com/labs/dir/sage/) / [项目落地页](https://nvlabs.github.io/sage/)
- **项目页 / 代码：** 已开源 — 官方代码仓库 [NVlabs/sage](https://github.com/NVlabs/sage)（Apache-2.0；已发布三维场景生成、数据增强、Isaac Lab 动作数据采集及 Robomimic 策略训练代码）；配套数据集 [nvidia/SAGE-10k](https://huggingface.co/datasets/nvidia/SAGE-10k)（10,000 个交互式室内场景、50 种房间类型与风格、565K 个独立生成的 3D 物体）
- **核对范围：** 通读 `arXiv:2602.10116v2` HTML 全文（Abstract、§1 Introduction、Table 1、§2 Related Works、§3 Method §3.1–§3.2、§4 Experiments §4.1–§4.2、Table 2–5、Figure 1–11、§5 Conclusion、References [1]–[68]、Appendix A A.1–A.3 与 Appendix B B.1–B.3），并核对项目页跳转落地页与官方仓库 [NVlabs/sage](https://github.com/NVlabs/sage) 的文档与核心源码入口。
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[SAGE: Scalable Agentic 3D Scene Generation for Embodied AI](../../wiki/entities/paper-agentic-2602-10116.md)

### 1. 问题定义与核心框架

- **定位：** 面向具身智能（Embodied AI）的智能体化三维场景与训练数据生成框架，输入开放词汇（open-vocabulary）文本任务描述（并可扩展至参考图像或铰接物体检索），自动输出可直接部署在物理仿真器中的交互式 3D 环境，以及用于模仿学习的机器人动作轨迹数据集。
- **MCP 架构（§3.1 / Appendix B.1.1）：** 基于 Model Context Protocol（MCP），智能体作为 MCP Client，场景生成工具与物理仿真器托管在 MCP Server 后端，工具输入参数与输出均格式化为 JSON 字典字符串。智能体根据反馈动态选择下一步调用的工具，无硬编码固定执行图。
- **四大生成器（§3.1.1 / Appendix B.1.2）：**
  1. `Scene Initializer`：LLM 预测房间类型、尺寸（及多房间连通门窗布局），用 MatFuse（附录亦支持 FLUX.1-dev）生成地板与墙体材质，并输出包含文本描述、物理属性与约束的对象提案列表。
  2. `Asset Placer`：使用 TRELLIS 根据文本生成 3D 物体网格并执行减面、非流形修复与补洞（水密处理），由 VLM 估计高度、质量、金属度与粗糙度等 PBR/物理参数；LLM 将放置关系分为 `floor`、`wall`、`on-top` 三类，前两者采用网格采样 + 深度优先搜索（DFS）避碰，`on-top` 通过支撑网格表面法向采样支持多层支撑图。
  3. `Asset Mover`：根据视觉评论器反馈定位对象、临时移除后复用放置规划器重定位；若空间不足则恢复原位并上报。
  4. `Asset Remover`：根据评论器反馈定位并删除多余或冲突对象。
- **双评论器闭环（§3.1.2 / Appendix B.1.3）：**
  - `Visual Critic`：输入当前场景配置与多视角渲染（俯视图 + 四个角落视角），检查自然物体组合、稀疏补全、任务关键物体是否齐全及错位调整建议。
  - `Physics Critic`：将场景加载至 Isaac Sim（墙面物体设为静态，其余设为刚体动态）检测重力与碰撞下的稳定性。若初次模拟后失稳（如枕头立在床上倒下），记录落定后位姿并二次仿真复核；全部候选均失败则上报建议更换更小物体或更换支撑面。

### 2. 场景缩放与具身动作数据合成（§3.2 / Appendix B.2–B.3）

- **三级场景增强：**
  1. 对象位姿级（Object Configuration-level）：重采样任务相关物体在场景内的摆放位姿；
  2. 对象类别级（Object Category-level）：用 LLM 改写任务相关物体的形状、颜色、材质等文本描述，再用 TRELLIS 重新生成保持同类语义的多样 3D 资产（含墙地纹理增强）；
  3. 场景布局级（Scene Layout-level）：保留已生成的任务相关物体并按新布局重定位，其余房间几何与任务无关背景由智能体重新生成。每次增强后均复跑 Physics Critic。
- **机器人动作生成与策略训练：**
  - Pick-and-Place（Franka Emika Panda）：从深度渲染用 M2T2 生成抓取候选位姿，通过 CuRobo 网格碰撞检测与 IK 规划无碰抓取-抬升-放置（简化为释放丢落）轨迹。
  - Mobile Manipulation（Franka Emika Panda + Omron LD-60 移动底盘）：在 2D 占据栅格上用 RRT 规划双向碰撞自由导航路径，结合 M2T2 + CuRobo 执行抓放；长时程任务拆分为 4 个顺序子策略（各附阶段终止信号）。
  - 使用 Robomimic 训练基于多视角 128×128 RGB-D 与末端轨迹的 Diffusion Policy。

### 3. 关键实验数据（均经全文核对）

- **场景生成主对比（Table 2，每类房间 10 场景均值；视觉由 GPT-4.1 评分，碰撞用 `trimesh`，稳定性在 Isaac Sim 仿真 120 步后判位移 $\le 0.2\text{ m}$ 且旋转 $\le 8^\circ$）：**
  - 平均指标（Bedroom / Kitchen / Living Room 均值）：Holodeck 为 `#Obj 30.3 / Real. 7.5 / Func. 6.7 / Lay. 5.0 / Comp. 6.5 / Coll. 22.0% / Stab. 63.8%`；SceneWeaver 为 `24.4 / 8.6 / 8.9 / 7.3 / 7.2 / 32.8% / 67.7%`；SAGE 为 `48.2 / 8.8 / 9.5 / 7.9 / 8.2 / 1.9% / 99.9%`（其中 Kitchen 与 Living Room 稳定性达 `100.0%`，Bedroom 为 `99.8%`）。
- **评论器消融（Table 3，三类房间各 5 场景均值）：** 无评论器 `#Obj 35.3 / Coll. 7.8% / Stab. 80.3%`；仅 Visual Critic 提升物体数至 `50.1`、碰撞降至 `3.7%`、稳定 `84.1%`；仅 Physics Critic 将碰撞降至 `1.9%`、稳定升至 `99.6%`；两者联合达 `#Obj 53.7 / Coll. 0.8% / Stab. 100.0%`。
- **SAGE-10k 与扩展能力（§4.1.2–§4.1.3）：** 预生成 10,000 场景（50 种房间类型与风格，565K 独立 3D 物体）；可自然扩展至多房间连通布局、Qwen3-VL 图像条件生成（语义一致但非像素对齐）以及从 PartNet-Mobility 检索铰接物体并规划开关抽屉轨迹。
- **具身策略学习与交叉评测（§4.2，Table 4–5，3 个留出测试场景 × 90 次随机初始化）：**
  - 数据规模：Pick-and-Place 生成超 `28k` 条演示（264 个独立物体，每 GPU 8 并行环境）；Mobile Manipulation 生成近 `50k` 条演示（50 个多样场景，每 GPU 2 并行环境）。
  - Table 4（特权运动规划 Agent vs. 视觉 Diffusion Policy Rollout）：Pick-and-Place 特权 Agent 训练 `65.3%` / 测试 `57.7%`，Policy Rollout 训练 `63.4%` / 测试 `50.0%`；Mobile Manipulation 特权 Agent 训练 `68.4%` / 测试 `52.8%`，Policy Rollout 训练 `54.6%` / 测试 `46.0%`。
  - Table 5（Mobile Manipulation 跨源泛化成功率 %）：在 Baseline 1（SceneWeaver 风格）训练并在 Baseline 1 / Baseline 2 / SAGE 测试分别为 `13.2 / 9.3 / 14.4`；Baseline 2（Holodeck 风格）为 `13.5 / 16.2 / 13.1`；SAGE 训练的策略分别为 `39.1 / 24.7 / 46.0`，跨分布依然大幅领先。
- **运行耗时（Appendix A.3，NVIDIA L40S）：** 单卡 TRELLIS 约 `15 s`/物体（8 卡并行 `2–3 s`/物体）；Isaac Sim 单次候选稳定性验证 `1–2 s`（采用早停策略）；生成含 20 个物体的场景总计约 `10 min`；Isaac Lab 动作数据采集并行均摊后 Pick-and-Place 约 `2–3 s`/条、Mobile Manipulation 约 `8–10 s`/条；Diffusion Policy 收敛需数小时。

### 4. 局限与边界

- 当前主力评测集中在室内刚体场景，室外环境、全场景铰接体生成及可变形物体仍待拓展。
- 动作生成目前聚焦抓放（放置简化为释放丢落）与移动导航的组合，尚未覆盖精细装配或真机闭环部署（仅在仿真中验证策略扩展趋势）。
- 依赖多组大型基础模型（如 `gpt-oss-120b`、`Qwen3-VL-30B-A3B-Instruct` / `Qwen3-VL-32B-Thinking`）与 TRELLIS，生成单场景仍需约十分钟级计算时间。
