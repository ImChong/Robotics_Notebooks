# Kevin Zakka｜机器人仿真与 Sim2Real 研究主页（一手资料归档）

- **类型：** 研究者个人主页 / 论文及开源软件索引（非单篇论文、非统一代码仓库）
- **官方主页：** <https://kzakka.com/>
- **官方 GitHub：** <https://github.com/kevinzakka>
- **核查日期：** 2026-11-17
- **核对范围：** 通读主页简介、News、Research、Publications、Software、Teaching、Technical Writing、Awards、Industry Experience、Contact 等公开正文及对应入口；通读 2026-11-16 的 [Closing the loop](https://kzakka.com/closing-the-loop/) 技术文章（从模型假设、踝部机械限位与闭链传动，到重置、积分器、选型与模型下载）；查验主页链接的 mjviser、mjlab、MuJoCo Playground、MuJoCo Menagerie、Mink 官方 GitHub 代码入口。所称“通读”仅指**个人主页和该篇博客**，不是其全部论文或代码的逐一深读。
- **通读：** 已完成
- **对应 Wiki：** [Kevin Zakka｜人形 Sim2Real 与 MuJoCo 工具生态](../../wiki/entities/kevin-zakka.md)

## 身份与研究主线（主页自述）

- UC Berkeley Robot Learning Lab 博士生，导师 Pieter Abbeel；主页称其为 final-year PhD student，Google PhD Fellowship 资助。
- 三条主要研究方向：**Humanoid Whole Body Control**（人形全身移动操作）、**Data Priors**（人体示范/动捕先验）、**Simulation**（规模化仿真与系统辨识支撑 Sim2Real）。
- 主页列出 2025 年 Boston Dynamics Electric Atlas 团队实习经历，以及 Google DeepMind 仿真团队实习经历。身份与时间为**2026-11-17 主页快照**，后续可能变动。

## 主页可核验研究成果

| 主页列出工作 | 原始入口 / 发表状态（主页所列） | 本库关系 |
|---|---|---|
| mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning | [arXiv:2601.22074](https://arxiv.org/abs/2601.22074)，2026 预印本；[代码](https://github.com/mujocolab/mjlab) | [mjlab](../../wiki/entities/mjlab.md) |
| MuJoCo Playground | [arXiv:2502.08844](https://arxiv.org/abs/2502.08844)，RSS 2025 Outstanding Demo Paper Award；[代码](https://github.com/google-deepmind/mujoco_playground) | [MuJoCo Playground](../../wiki/entities/mujoco-playground.md) |
| ALOHA 2 | [arXiv:2405.02292](https://arxiv.org/abs/2405.02292)；[代码](https://github.com/tonyzhaozh/aloha) | 双臂遥操作硬件 |
| RoboPianist | [项目演示](https://kzakka.com/robopianist/)，CoRL 2023 | 双灵巧手 + 强化学习钢琴基准 |
| XIRL | [官方项目页](https://x-irl.github.io/)，CoRL 2021 Best Paper Finalist | 跨本体逆强化学习 |
| Form2Fit | [官方项目页](https://form2fit.github.io/)，ICRA 2020 Best Paper Finalist | 拆解示范驱动的装配学习 |

注：此表记录**主页所列作者作品和一手入口**，不等于逐篇论文方法、实验和最新版本均完成核实。

## 开源软件入口与边界

主页「Software」列有独立 GitHub 仓库，确认对应公开代码入口：

- **仿真训练：** [mjlab](https://github.com/mujocolab/mjlab)、[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- **可视化：** [mjviser](https://github.com/mujocolab/mjviser)
- **模型与机器人描述：** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)、[robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py)
- **控制工具：** [Mink](https://github.com/kevinzakka/mink)（微分 IK）、[mjctrl](https://github.com/kevinzakka/mjctrl)（简洁控制器示例）

**开源结论：** 上述软件有独立的公开仓库；个人主页本身是作品导航而非可运行的统一框架。不能据此把所有列出的论文、训练数据、硬件实测与权重一概判为已公开。

## 最新技术写作入口（2026-11-16）

[Closing the loop: How to model the Unitree G1's push-rod ankle in MuJoCo](https://kzakka.com/closing-the-loop/)（Kevin Zakka、Qiayuan Liao，2026-11-16）完整介绍：

1. 以实体测量与 fixed tendon 不等式表达脚踝 pitch/roll 的**耦合限位**；串联双关节矩形范围不能表示真实机械止挡。
2. 以 spatial tendon + equality constraint 或显式刚体连杆复建**推杆闭链**，传动雅可比把电机力矩和转动惯量映射到踝部。
3. 重置时使电机与脚踝状态满足闭合约束；论文式追踪实验中，文章对比了不同电机侧增益映射的失效率。
4. 文中指出在其测试配置下，直接对 pitch/roll 施加位置执行器时 `implicitfast` 大步长可能不稳定，而 `discrete` 更适用于该闭链结构；文章提供 [G1 踝部模型文件](https://kzakka.com/closing-the-loop/) 下载。

以上是**该博客的实测范围与结论**，不能泛化到所有 MuJoCo 模型或真实机器人。文章的示例模型可以下载，但本次没有验证独立完整训练项目或真机部署代码。

## 归档与复用边界

- 官网 footer 标注“Last revised: April 2026”，但 News 已增加 2026 年 11 月文章；**不能用 footer 推断最新论文发表日期**。
- 本次收录的对象是**研究者 / 软件导航节点**；独立论文或代码仓库应继续维护各自已有详情节点，不重复造 paper 实体。
- 个人网站内容版权标注 CC BY-NC-SA 4.0；此处仅保存归纳与引用，不镜像全文。

## 对 Wiki 的映射

- [Kevin Zakka 导航实体](../../wiki/entities/kevin-zakka.md)
- [Qiayuan Liao 研究索引](../../wiki/entities/qiayuan-liao.md)
- [mjlab](../../wiki/entities/mjlab.md) · [mjviser](../../wiki/entities/mjviser.md) · [Mink](../../wiki/entities/mink-ik.md)

## 参考链接

- <https://kzakka.com/>
- <https://kzakka.com/closing-the-loop/>
- <https://github.com/kevinzakka>
