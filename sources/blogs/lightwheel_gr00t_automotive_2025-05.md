# From Sim to Factory: How Lightwheel Deployed GR00T N1 Humanoids in Automotive Production

> 来源归档（blog / Lightwheel 官方 LinkedIn 文章）+ NVIDIA COMPUTEX 2025 新闻稿中的 Lightwheel 提及

- **标题：** From Sim to Factory: How Lightwheel Deployed GR00T N1 Humanoids in Automotive Production
- **类型：** blog（发布于 LinkedIn Pulse，官网 Blogs 列表外链）
- **作者：** Steve Xie, Ph.D.（署名 Founder and CEO of Lightwheel）
- **原始链接：** <https://www.linkedin.com/pulse/from-sim-factory-how-lightwheel-deployed-gr00t-n1-humanoids-ryycc>
- **发表日期：** 2025-05-19（LinkedIn 显示 Published May 19, 2025；官网列表同）
- **入库日期：** 2026-10-10
- **抓取方式：** curl 读取 LinkedIn 公开页 HTML（未登录可读全文）后去标签核对
- **覆盖 wiki：** [Lightwheel 公司页](../../wiki/entities/lightwheel.md)「GR00T N1 汽车工厂部署」小节

## 核心摘录（归纳，非全文）

### 场景与结论（自报）

- 与 NVIDIA 合作，把 **Isaac GR00T N1** 部署到 Lightwheel 的人形机器人上，进入 **吉利（Geely）在产汽车工厂**；具体场景为 **发动机厂质检** 工位：把零件放入指定料箱、转运到指定上料架、双臂搬运大件 / 重件、在有人和移动设备的环境中做情境感知运动。
- 机器人为 **Unitree H1**（文中称 H1 的「33-DOF joint structure」，推测含灵巧手自由度）；推理跑在 **NVIDIA GeForce RTX 4090**（原型期优先迭代速度，尚未做嵌入式 / 板载部署）。
- **边界（作者自述）：** 尚未达到生产认证级完全自主；「在监督条件下的早期部署」表现「consistently reliable」，**无成功率、节拍或时长数据**。

### 技术栈

| 环节 | 做法（自报） |
|------|--------------|
| 仿真平台 | **Lightwheel Simulation Platform**：云原生，含弹性算力、混合仿真引擎、可定制 benchmark 层（支持 RoboCasa、Behavior1K 与 Lightwheel SimReady 资产） |
| 混合仿真 | **Lightwheel Hybrid Sim**：Isaac Sim 负责光追渲染 / 传感器 / 可视化，MuJoCo 负责底层物理与稳定接触 |
| 数字孪生 | 用 NVIDIA Omniverse + SimReady 资产管线复刻产线（传送带、料箱、工具、检测台、光照） |
| 遥操作采数 | Apple Vision Pro / Meta Quest 在仿真中遥操作 Unitree H1（灵巧手分拣托盘内圆柱零件、双臂抬重托盘） |
| 虚实配比 | 仿真 : 真实 = **100 : 1** 共训（真实样本用同一 VR 方案在实体 H1 上采集） |
| 训练样本 | RGB、关节 / 末端位置速度、GPT 生成的任务语言描述、场景元数据 |
| 数据扩增 | **DexMimicGen** 跨场景泛化轨迹 + Isaac Sim 内光照 / 材质 / 位置 / 杂乱度随机化 |
| 数据 QA | 两阶段：自动校验（视觉 / 物理真实感、标注完整性、模态对齐）+ 人工复核；称经验来自自动驾驶合成数据 |
| 本体适配 | GR00T N1 原主要在 Fourier GR1 上训练；对 **System 2**（视觉-语言规划）用工厂特定提示微调，对 **System 1**（扩散动作生成）重配 H1 关节结构、限位与电机模型 |

### 后续方向（作者列举）

可变形物体（线缆、织物、软材料）进入 Isaac Sim 的 **Newton** 后端；家庭场景 SimReady 资产并验证 Newton 兼容；碰撞分解与网格优化以提升 RL 吞吐；用 GR00T N1 作为仿真中的半自主示范者再由人工审校；作为 GR00T Nx 早期采用者并为其训练贡献合成示范。

## NVIDIA 新闻稿中的 Lightwheel 提及（2025-05-18）

- **链接：** <https://nvidianews.nvidia.com/news/nvidia-powers-humanoid-robot-industry-with-cloud-to-robot-computing-platforms-for-physical-ai>（COMPUTEX 2025，发布 GR00T N1.5 与 GR00T-Dreams）
- 副标题把 Lightwheel 列入「采用 NVIDIA Isaac 的机器人公司」（与 Agility、Boston Dynamics、Foxconn、NEURA、XPENG Robotics 并列）。
- 正文：GR00T N 系列早期采用者包括 AeiRobot、Foxlink、**Lightwheel**、NEURA；「Lightwheel is harnessing them to validate synthetic data for faster humanoid robot deployment in factories」。
- 新闻稿 **未** 提吉利、H1 或部署结果；工厂部署细节只来自 Lightwheel 自述。

## 对 wiki 的映射

- [Lightwheel 公司页](../../wiki/entities/lightwheel.md)
- [Isaac GR00T](../../wiki/entities/isaac-gr00t.md)、[GR00T N1.5](../../wiki/entities/paper-gr00t-n1-5.md)
- [DexMimicGen](../../wiki/entities/paper-notebook-dexmimicgen-automated-data-generation-for-bimanu.md)

## 可信度与使用边界

- 公司自述案例：客户名（吉利）只出现在 Lightwheel 文章中，未见吉利或 NVIDIA 独立确认；无定量指标。
- 「100:1 虚实配比」是该项目的做法描述，不是通用最优比例的实验结论。
