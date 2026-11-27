# GoalVLM: VLM-driven Object Goal Navigation for Multi-Agent System

- **作者：** MoniJesu James, Amir Atef Habel, Aleksey Fedoseev, Dzmitry Tsetserokou
- **机构：** Intelligent Space Robotics Laboratory, Center for Digital Engineering, Skolkovo Institute of Science and Technology (Skoltech), Moscow, Russia
- **版本 / 载体：** arXiv:2603.18210v1 [cs.RO]（2026-03-18 提交，8 页，5 图，CC BY 4.0）
- **原文：** https://arxiv.org/abs/2603.18210（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2603.18210）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv 原文及检索来源未提供独立项目主页或官方 GitHub/代码仓库链接）
- **核对范围：** 通读 arXiv:2603.18210v1 全文（Abstract、Sections I–V、Algorithm 1、Tables I–IV、Figures 1–5 题注及 References），并核对检索元数据（`context.json` 中 `project_page_status=0`、`github_urls=[]`、`candidate_repos=[]`）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[GoalVLM 实体页](../../wiki/entities/paper-agentic-2603-18210.md)

1. **研究问题与形式化设定（Sections I–III-A）**
   - 现有目标导航（ObjectNav）与多智能体协同导航大多局限于封闭词表或依赖固定类别集的预计算概率图，难以在测试时泛化到自由文本指定的开放词表目标。
   - 论文将多智能体开放词表目标导航形式化为 Dec-POMDP：$N$ 个智能体在未知环境 $\mathcal{E}$ 中定位自然语言类别 $g \in \mathcal{G}$（覆盖 379+ 类别）的目标实例。
   - 单步观测 $o_t^i = (I_t^{rgb}, I_t^{depth}, p_t^i)$，全局位姿 $p_t^i=(x,y,\theta)$；离散动作空间包含 6 个动作：`STOP`、`MOVE_FORWARD`（0.25 m）、`TURN_LEFT` / `TURN_RIGHT`（$\pm 30^\circ$）、`TURN_LEFT_S` / `TURN_RIGHT_S`（$\pm 10^\circ$）。
   - 按 GOAT-Bench 协议，每回合包含 5–7 个顺序开放词表子任务；智能体在目标欧氏距离 $d_{success}=1.0\text{ m}$ 内执行 `STOP` 即判该子任务成功，单子任务步数预算为 500 步。

2. **核心方法流水线（Sections III-B–III-H, Algorithm 1）**
   - **以自我为中心的 BEV 语义建图（Sec. III-B）**：针对竖屏传感器（$360\times 640 \to 160\times 120$）非均匀缩放带来的纵向畸变，采用双焦距内参 $f_x = f_s \cdot (W_p/W_s)$、$f_z = f_s \cdot (H_p/H_s)$ 反投影深度点云，经俯仰角旋转 $\mathbf{R}_\phi$ 与传感器高度 $h_s=1.31\text{ m}$ 平移至地理中心坐标系，再通过可微体素泼溅（Voxel Splatting）生成体素网格 $\mathcal{V} \in \mathbb{R}^{V\times V\times Z\times C}$，并在可行走高度带 $[25\text{ cm}, h_s + 50\text{ cm}]$ 上切片统计障碍物栅格。
   - **双路零样本视觉-语言感知（Sec. III-C）**：
     - **开放词表检测与定位**：采用 SAM3 根据文本查询 $g$ 生成检测框 $\{(b_i, c_i)\}$ 与像素级掩码 $\mathcal{S}_g$；通过 `GoalProjector` 将掩码经标定深度反投影至 BEV 语义地图，并使用多视角确认（连续 $N_{confirm} \ge 2$ 帧置信度 $c_i > \tau_{det}=0.30$）抑制误检。
     - **空间常识推理**：调用 SpaceOM（论文文本称 SpaceOM，参考文献 [3] 引 SpatialVLM），通过结构化提示链（场景描述、房间类型分类、感知门控、多边界排序）输出候选边界（frontier）包含目标的先验概率 $P(g \mid f_i, \mathcal{H}_t)$。
   - **约束引导边界选择与贝叶斯价值图（Sec. III-D, III-F）**：提取至多 $K=4$ 个探索边界，结合 VLM 概率评分 $s_i^{\text{vlm}}$ 与贝叶斯价值图归一化 UCB 分数 $\hat{v}_i$（权重 $w=0.35$，探索系数 $\beta=1.7$，初始先验 $\mu_0=0.5, \sigma_0^2=0.5$）：$U(f_i) = (1-w)\cdot s_i^{\text{vlm}} + w\cdot \hat{v}_i$。价值图在深度可见锥掩码 $m(x,y)$ 内以观测方差 $\sigma_{obs}^2 = 1-m(x,y)$ 进行贝叶斯融合更新。
   - **FMM 局部路径规划（Sec. III-E）**：在可通行栅格上求解 Eikonal 方程 $|\nabla T(\mathbf{x})|=1/v(\mathbf{x})$ 计算测地距离场（复杂度 $O(n\log n)$），通过梯度下降提取并截断至 25 格（1.25 m）内的短期目标（STG），映射为离散转向与前进动作。
   - **去中心化多智能体协同（Sec. III-G）**：部署 $N=2$ 个同构智能体（$h=1.31\text{ m}$、$\text{HFOV}=42^\circ$、深度范围 $[0.5\text{ m}, 5.0\text{ m}]$）。每步通过逐元素最大池化 $\mathcal{M}_{global}(x,y,c)=\max_i \mathcal{M}_t^i(x,y,c)$ 融合局部语义地图；边界分配采用顺序贪心协议，$a_0$ 选定最高分边界后将其移出候选集，$a_1$ 再从剩余边界中选择，避免重复探索。

3. **仿真评测、消融与真机验证（Section IV, Tables I–IV）**
   - **GOAT-Bench `val_unseen`（Tab. I）**：在 HM3D 场景 357/360 个回合（1032 个顺序子任务）上，GoalVLM（$N=2$，零样本多智能体）取得 **55.8% 子任务 SR** 与 **18.3% SPL**，超过 SenseAct-NN Skill Chain（29.5% SR / 11.3% SPL）、Modular GOAT（29.4% SR / 17.5% SPL）、3D-Mem（28.8% SR / 15.8% SPL）、DyNaVLM（25.5% SR / 10.2% SPL）与 VLMNav（16.3% SR / 6.6% SPL），逼近需 3B 参数 VLM 端到端微调的 AstraNav-Memory（62.7% SR / 56.9% SPL），但 SPL 明显偏低。
   - **消融实验（GOAT-Bench `val_seen` 50 回合 / 137 子任务，Tab. II）**：完整 GoalVLM 为 **56.2% SR / 16.3% SPL / 1.37 m DTG**；移除 VLM 推理降至 **54.0% SR / 15.0% SPL / 1.94 m DTG**；退化为单智能体（$N=1$）降至 **38.6% SR / 15.0% SPL / 2.00 m DTG**（IV-C3 记作约 $1.5\times$ 提升，IV-F 与 V 节文本另述及 32.0%→56.2% / $1.8\times$，wiki 撰写时以 Tab. II 实测表为准并标注文本差异）。另外对比了双焦距内参修正（$f_x \neq f_z$）对消除地面像素误投障碍物层的关键作用。
   - **失败归因与按目标/位置统计（Tab. III, Fig. 4–5）**：357 回合中完美成功 111 个（31.1%）、部分成功 172 个（48.2%）、完全失败 74 个（20.7%）。74 个完全失败按终局 DTG 分布为：检测/定位失败（1.5–3.0 m）43.7%、探索失败（>3.0 m）36.6%、接近/停止失败（<1.5 m）19.7%。镜面与玻璃等反光/透明物体表现最差（`mirror` 47.6%, $n=63$；`window glass` 33.3%, $n=9$），大件典型物体较高（`refrigerator` 72.7%, $n=55$；`piano` 80.0%, $n=5$）。各顺序位置子任务 SR 相对稳定（第 1–4 位分别为 56.1%、58.8%、55.6%、49.0%）。
   - **真机初步验证（Sec. IV-G, Tab. IV）**：使用 2 架定制四旋翼无人机（Intel RealSense D435/D435i、Orange Pi 5、ArduPilot/MAVROS ROS 2、Vicon 动捕）在 $10\times 5\text{ m}$ 室内场地通过 ZMQ（20 Hz 状态 / 10 Hz 图像）向地面站推流，验证了 SAM3 与深度反投影 BEV 建图对椅子、手推车、行李箱等目标的实时感知定位；闭环自主 VLM 边界导航留作未来工作。

4. **核心局限（Sections IV-F, IV-G, V）**
   - 2D BEV 地图无法表达多层建筑或垂直堆叠物体；
   - 模块化边界探索导致 SPL 显著低于端到端微调记忆策略（18.3% vs. 56.9%）；
   - SAM3 在镜面/透明表面易出现虚警或深度反射穿透墙体定位偏差，且对小物体/遮挡物体召回不足；
   - 真机实验依赖 Vicon 外部动捕提供里程计真值，尚未完成机载 SLAM 与闭环自主多机飞行导航验证。
