# Bridging Semantics and Physics with Constrained LLMs for Safe and Trustworthy Robotic Manipulation

> 来源归档（paper）

- **作者：** Wenhao Hong, Lan Wei, Dandan Zhang（通讯作者：`d.zhang17@imperial.ac.uk`）
- **机构：** Imperial College London, London, UK
- **版本 / 载体：** arXiv:2608.29379v1 [cs.RO]（2026-08-29 提交；arXiv 元数据备注为 `ECCV Workshop paper`）
- **原文：** <https://arxiv.org/abs/2608.29379>（arXiv 摘要页）、<https://arxiv.org/html/2608.29379>（实际通读的 arXiv HTML 全文）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27 核查 arXiv 页面与全文：未提供独立项目主页或本文官方源码仓库；原文参考文献 `[16]` 所引 GitHub 链接 `https://github.com/ultralytics/ultralytics` 为第三方依赖检测器 Ultralytics YOLOv8，并非本论文实现仓库）
- **核对范围：** 通读 arXiv HTML 全文（`paper_text.txt`，共 451 行、46,514 字符）：Abstract、§1 Introduction 与 Contributions、§2 Related Work（§2.1–§2.3）、§3 Method（§3.1–§3.4，含公式 (1)–(2)、Table 1、Fig. 1–3 图注）、§4 Experiments（§4.1–§4.6，含 Table 2–3、Fig. 4–6 图注）、§5 Conclusions and Future Work 与全部 35 条参考文献；同步核对 `context.json` 中候选仓库状态（确认无本文官方源码仓，无需创建 `sources/repos/` 归档）。
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与核心契约（§1, §3.1）

- 针对家庭/厨房杂乱环境中自然语言引导操作的**语言-动作鸿沟（language-action gap）**：LLM 能在语义上生成看似合理的规划，但可能因不满足运动学可达性、关节限位、碰撞约束或非法物体引用而在物理上不可行。
- 区别于基于学习世界模型（learned world models）预测动作后果的路线（部分可观测下易幻觉出违反物理一致性的预测），本文采用显式几何建模与**类型化推理-执行契约（typed contract）**：将上层语义推理与下层运动执行严格解耦。

### 2. 三层系统架构（§3.2–§3.4）

1. **Semantic Layer（语义层，§3.2）：**
   - 推理骨干采用 **Claude Sonnet 4**，仅接收自然语言指令与感知落地后的结构化场景上下文（物体标识符、语义类别、机器人状态），不直接处理原始图像，也不生成位姿或轨迹。
   - **状态抽象（公式 1）：** $S = \{\mathrm{id}_{\mathrm{src}}, \mathrm{id}_{\mathrm{tgt}}, B_{\mathrm{pour}}\}$，仅绑定源容器 ID、目标容器 ID 与是否倾倒的布尔标志。
   - **双轨策略（Dual-track policy）：**
     - **澄清轨（Clarification track）：** 面对指代歧义或欠定指令（例如桌面存在两个杯子），调用受限的 ReAct 循环，仅允许查询感知或向用户追问澄清，**禁止**调用任何驱动工具或下发运动指令；若歧义未解则中止任务。
     - **确定性执行轨（Deterministic execution track）：** 对明确指令直接选用预验证的有限状态机（FSM）模板，按规范动作序（`pick` → `move` → `pour` → `place`）推进。
2. **Protocol Layer（协议层，§3.3）：**
   - 基于 **Model Context Protocol (MCP)** 将机器人能力暴露为有限集合的 JSON Schema 工具，例如倾倒工具（公式 2）：$\mathrm{Tool}_{\mathrm{pour}}: (\mathrm{id}_{\mathrm{tgt}} \in S_{\mathrm{scene}}, \theta_{\mathrm{tilt}} \in [0, \theta_{\max}], \tau_{\mathrm{duration}} \in \mathbb{R}^+)$。
   - **运行时三阶段校验：**（1）Schema 校验（必填字段与数据类型）；（2）范围校验（如倾角上限、持续时间在允许物理区间内）；（3）指代校验（`id` 必须存在于当前已注册的感知场景物体集合 $S_{\mathrm{scene}}$ 中）。任一校验失败即在进入规划器前拒绝。
   - **原子到组合绑定（Table 1）：**
     - `setup_scene` → 感知注入 → 生成碰撞体
     - `pick(object)` → 抓取生成 → IK 过滤 → 接近 → 闭合 → 抬起
     - `move_to_pour` → 笛卡尔规划 → 倾斜约束检查 → 速度缩放 → 执行
     - `place(target)` → 放置位姿生成 → 接近 → 张开 → 撤回
   - **可审计与后端解耦：** 记录所有语义调用参数，将底层失败规范化为 `PlanningFailed`、`TargetUnreachable` 等结构化错误码，先绑定至运动学验证环境做预检，再绑定至 UFactory 850 物理驱动。
3. **Execution Layer（执行层，§3.4）：**
   - **硬件与标定：** UFactory 850 机械臂（配二指夹爪）+ 眼在手外（eye-to-hand）Intel RealSense D435i RGB-D 相机；通过 $5 \times 7$ ChArUco 标定板估计外参（亚厘米级平移精度、$1^\circ\text{--}2^\circ$ 旋转精度）。
   - **动态场景落地：** RGB 经 YOLO 检测器（Ultralytics YOLOv8）结合类别置信度阈值识别后，在包围盒内对对齐深度取中值滤波去噪，反投影并转换到机器人基座系，通过 `setup_scene` 以**单位尺度预定义圆柱代理体（cylindrical proxies）**而非在线重建网格注入 MoveIt 规划场景。
   - **约束运动生成与 Verify-then-act：** 基于 **MoveIt Task Constructor (MTC)** 将技能分解为 Generators（抓取候选与 IK 过滤）、Propagators（安全高度接近、下降、抬起及速度/加速度/姿态约束）与 Connectors（连接子状态并在失败时回溯）。倾倒采用保守分段倾斜-保持（tilt-and-hold）剖面。所有候选轨迹在发送真机前必须在重建规划场景中完成连续性、可达性与无碰撞回放验证；高风险阶段间（如抓后倾倒前）插入预定义的无碰撞**安全位姿（safe pose）**以抑制误差累积并支持恢复。

### 3. 真机实验与消融（§4, Table 2–3）

- **实验条件：** UFactory 850 真机评测，每任务/每配置 $N=10$ 次独立试验（单次占 10 个百分点）。
- **四项倾倒任务（Table 2）：**
  - `T1: Single Pour`：Ours **80%** / FSM/BT **90%** / LLM w/o task graph **60%**
  - `T2: Multi-Source Pour`：Ours **70%** / FSM/BT **40%** / LLM w/o task graph **30%**
  - `T3: Multi-Material Pour`（水、咖啡豆、鸡蛋等离散固体）：Ours **80%** / FSM/BT **60%** / LLM w/o task graph **60%**
  - `T4: Complex Scene`：Ours **60%** / FSM/BT **10%** / LLM w/o task graph **40%**
  - **均值（Mean）：** Ours **72.5%** / FSM/BT **50.0%** / LLM w/o task graph **47.5%**
- **跨技能泛化（§4.4）：** 迁移至水果抓取放置（apple/orange → bowl），仅新增任务级 tool schema 与 MTC 模板，复用同一规划、协议与 verify-then-act 栈，10 次试验取得 **90%**（9/10）成功率；唯一失败源于相机视野边缘附近位姿估计误差导致接近阶段碰撞。
- **恢复机制消融（Table 3，保留 schema 校验与 verify-then-act）：**
  - 移除 `safe-pose reset`：T1 50% / T2 20% / T3 70% / T4 20%，均值降至 **40.0%**（相对完整系统 72.5% 下降 32.5 个百分点）。
  - 移除 `retry`：T1 20% / T2 40% / T3 10% / T4 30%，均值降至 **25.0%**（下降 47.5 个百分点）。
  - 对比移除 `task graph`（Table 2）：均值 **47.5%**（下降 25.0 个百分点）。说明瞬时 IK/净空失败下的协议级恢复（retry 与 safe-pose）对真机端到端鲁棒性贡献最大。

### 4. 论文自述局限（§4.2, §5）

- 依赖重建规划场景的几何与位姿精度（Fig. 4 显示世界坐标系位姿接地误差存在不可忽视的长尾分布；近似外观容器仍可能出现语义误绑定；圆柱代理体存在模型-现实几何失配）。
- 未建模液体/颗粒材料动力学，也缺乏执行期视觉/触觉/听觉闭环反馈。
- 每条件仅 $N=10$ 次真机试验，单次波动占 10 个百分点，仅适合观察跨任务一致趋势。

## 对 wiki 的映射

- 对应实体页：[Bridging Semantics and Physics with Constrained LLMs for Safe and Trustworthy Robotic Manipulation](../../wiki/entities/paper-agentic-2608-29379.md)
