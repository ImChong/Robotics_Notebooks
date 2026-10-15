# Active Generative Simulation: Meta-Learning What to Simulate for Sim-and-Real Policy Co-Training（MetaGenSim，arXiv:2610.19479）

> 来源归档（ingest）

- **标题：** Active Generative Simulation: Meta-Learning What to Simulate for Sim-and-Real Policy Co-Training
- **缩写 / 框架：** **MetaGenSim**；问题定义 **Active Generative Simulation**
- **类型：** paper / sim-and-real co-training / generative simulation / meta-learning / data attribution
- **arXiv：** <https://arxiv.org/abs/2610.19479>（提交日期 2026-10-14；HTML：<https://arxiv.org/html/2610.19479v1>）
- **项目页：** <https://meta-gensim.github.io/>（CoRL 2026 学术页模板；页面 Paper / Code / arXiv 按钮）
- **代码：** 未发现官方代码。项目页 Code 按钮指向 `https://github.com/YOUR REPO HERE`（模板占位符），论文正文无代码发布声明。
- **作者：** Liqian Ma、Vaibhav Saxena、Utkarsh Aashu Mishra、Shuo Cheng、Yongxin Chen、Danfei Xu
- **机构：** Georgia Institute of Technology（项目页与论文一致）；资助：NSF Award 2409016
- **入库日期：** 2026-10-15
- **一句话说明：** 把仿真器的数据生成分布当作待优化对象：用探针数据集上的元梯度训练一个 rater 网络给每条仿真样本打效用分，再把效用映射到生成参数的密度上，定向生成仿真数据与少量真实演示协同训练策略。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** 唯一的 Code 按钮链接是模板占位符，未指向真实仓库；页面无模型或数据下载。
- **论文：** 未承诺代码或数据发布。
- **结论：** **未开源**（截至入库日未发现官方代码）。

## 摘录 1：问题与主张（Abstract，§1）

- **痛点：** 现有 sim-and-real 协同训练把仿真器当作固定数据源，靠均匀生成、离线检索或专家手动对齐参数；均匀生成会稀释目标任务相关数据。
- **主张：** 形式化为优化仿真器的数据生成分布，使协同训练后的真机表现最大；MetaGenSim 通过元学习 rater 自动发现高效用区域。
- **摘要数字：** 相对均匀生成，真机全任务成功率提升约 150%，与专家手动对齐相当；用轻量代理策略筛选的分布可迁移到 π0.5 的 VLA 微调；在任务部分对齐时可通过原子技能拼接与物体-容器组合利用数据。

**对 wiki 的映射：** 升格为 [`wiki/entities/paper-metagensim.md`](../../wiki/entities/paper-metagensim.md)；与 sim2real、域随机化、生成式数据增广等页面互链。

## 摘录 2：方法（§3–§4）

- **双层问题（§3.2）：** 上层选择生成分布 S_φ，下层为 `α·L(D_S) + (1−α)·L(D_R)` 的协同训练。
- **Rater 元学习（§4.1）：** 内层用 rater 给 sim/real 样本的 softmax 权重做加权 BC，外层在真实验证集上的损失对 rater 参数求梯度（通过 T 步内层展开）。
- **效用到分布（§4.2）：** 参数效用 U(φ) 为该参数生成轨迹上 rater 分数的平均；任务模式按指数效用采样；连续参数用加权核密度估计（Scott 带宽）。
- **探针数据：** 在参数空间上均匀采样一次，生成 D_probe；论文指出稀疏探针可能漏掉小的高效用区域。

**对 wiki 的映射：** 页面「核心原理」流程图为 探针 → rater 元学习 → 效用 → 分布 → 定向生成 → 协同训练。

## 摘录 3：实验（§5，Table 1、Table 2）

| 设置 | 论文数字 |
|------|----------|
| 真机（Franka Panda，4 任务 stackCube / mugBin / bowlTray / mugHang，每任务 10 条真实演示，策略为 Diffusion Policy，α = 0.9，每任务 20 次） | 全任务（Pick+Place）：MetaGenSim 58/80，Oracle 57/80，IWR 33/80，UniformGen 23/80，STRAP 27/80，RealOnly 26/80 |
| 仿真-仿真（MimicLabs 子集，8 任务，BC-RNN 与 DP，3 个种子） | 论文称与 Oracle 相当；具体曲线见 Fig. 3 与附录 G |
| π0.5 微调（Table 2） | 论文称代理策略筛选的分布可迁移；具体计数未在本次核对中提取 |

- 注：150% 相对提升与 Table 1 的全任务成功计数（58/80 对 23/80，约 +152%）一致。

**对 wiki 的映射：** 写入 `## 实验与评测`；π0.5 具体数字标注“未核实”。

## 对 wiki 的建议动作

- 新增 [`wiki/entities/paper-metagensim.md`](../../wiki/entities/paper-metagensim.md)，开源状态写「未开源（未发现官方代码）」，不写 `code:` 字段。

## 相关链接

- arXiv：<https://arxiv.org/abs/2610.19479>
- 项目页：<https://meta-gensim.github.io/>
- 文章摘录来源：[`sources/raw/wechat_embodied_station_robot_31_papers_2026-10-15.md`](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)
