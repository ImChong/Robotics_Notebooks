# Collision-Free Humanoid Traversal in Cluttered Indoor Scenes（Click-and-Traverse, arXiv:2601.16035）

> 来源归档（ingest · 原文通读）

- **标题：** Collision-Free Humanoid Traversal in Cluttered Indoor Scenes
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2601.16035>（v2，2026-01-23）
- **PDF：** <https://arxiv.org/pdf/2601.16035>
- **项目页：** <https://axian12138.github.io/CAT/>
- **代码：** <https://github.com/GalaxyGeneralRobotics/Click-and-Traverse>
- **机构：** Tsinghua University；Tongji University；Galbot；Shanghai Qi Zhi Institute；South China University of Technology；Peking University；Zhejiang University
- **作者：** Han Xue*, Sikai Liang*, Zhikai Zhang*, Zicheng Zeng, Yun Liu, Yunrui Lian, Jilong Wang, Qingtao Liu, Xuesong Shi, Li Yi†
- **核查日期：** 2026-11-21
- **核对范围：** 通读 arXiv:2601.16035v2 全文正文（§I Introduction 与 Table I、§II Related Works、§III Method A–C 与 Figs. 2–3、§IV Experiments A–C 与 Tables II–IV / Fig. 4、§V Conclusion）。
- **通读：** 已完成

## 核心技术与实验事实（原文核对）

- **问题定义：** 解决杂乱室内场景中的全空间（地面 $g$、侧向 $l$、头顶 $o$ 障碍物同时存在）且几何不规则条件下的无碰撞人形穿行（跨越、弯腰钻行、侧身挤过、绕行）。
- **Humanoid Potential Field（HumanoidPF，§III-A）：**
  1. **势场构建：** 吸引势 $U_{att}(x)=\eta \|x-g\|_{geo}$ 采用绕开障碍物的 3D 测地距离（而非欧氏距离），排斥势 $U_{rep}(x)$ 在有符号距离 $d(x)\le d_0$ 内按 $\frac{1}{2}\xi(1/d(x)-1/d_0)^2$ 递增；负梯度 $F=-\nabla U$ 经根节点优先权重 $w_0$（骨盆 $1.0$，其余 $0.5$）与碰撞紧迫度动态权重 $w_1(p_k)=\lambda\max(-\nabla d(x_k)\cdot v_k, 0.5)\exp(-d(x_k))$ 加权得到 $F^H$。
  2. **观测低通滤波：** 在 $K=13$ 个关键身体部位采样 $OBS_{Field}=\{F^H(x_k)\}$，连续势场自然滤除传感器局部几何噪声、缩小 Sim2Real 感知差距。
  3. **vMF 方向先验奖励：** 由 $F^H(x)$ 诱导每个部位运动方向 $\hat{v}_k$ 的 von Mises–Fisher（vMF）先验分布（均值 $\mu_k=F^H/\|F^H\|$、集中度 $\kappa_k=\kappa_{\max}\|F^H\|$），对数似然 $R_{Field}$ 提供前瞻性、跨场景免手工调参的稠密奖励。
- **混合场景生成与 Specialist-to-Generalist（§III-B）：**
  - 结合 **139 个 3D-FRONT 真实室内裁剪块**（$5\,\text{m}\times 5\,\text{m}$，半径 2 m 采目标点）与 **216 个程序化极端障碍场景**（长方体随机 $SO(3)$ 旋转 + 2D Perlin 噪声 + 3D 体素闭/开运算与难度课程）。
  - 每个场景用 PPO（32,768 并行环境）训专家策略，再用 **DAgger** 蒸馏成单一通才策略并注入传感器噪声与外力扰动。
- **真机 Click-and-Traverse（CAT）系统（§III-C）：** 在 **Unitree G1** 上部署基于 **Fast-LIO2 + OctoMap** 的 10 Hz LiDAR–IMU SLAM 建图与势场更新，用户在栅格地图上点击目标点即可自主穿行。
- **关键实验（§IV，MuJoCo + Unitree G1）：**
  - **Table II（8 类测试场景）：** 全场景 SR **86.6%–97.5%**（如 Hurdle-Crouch 93.9%、Side-Hurdle-Crouch 86.6%、Side-Crouch 96.9%），大幅高于 ASTraversal（0.5%–82.1%）与 Humanoid Parkour（0.4%–88.7%）；去掉 $OBS_{Field}$ 在 Side-Hurdle-Crouch 降至 53.7%，去掉 $R_{Field}$ 在 Side-Hurdle-Crouch 跌至 0.0%。
  - **Table III（30 个未见美术设计场景）：** 从仅 3D-FRONT（Easy 62.0% / Hard 1.2%）扩到 Base + Syn-Full-Hard 后零样本 SR 升至 **Easy 95.2% / Hard 66.7%**。
  - **Table IV（Unitree G1 真机 4 类杂乱场景，每类 5 次）：** HumanoidPF 达 CrouchPass **4/5**、HurdlePass **5/5**、SidePass **5/5**、CrouchHurdle **4/5**，高于体素网格（1–3/5）与多层高程图（1–3/5）。
- **原文自陈局限（§V）：** 当前框架尚未利用接触丰富交互（如倚靠障碍物或踏上支撑面）；面向完全未见、高度非结构化且极端杂乱场景的泛化仍具挑战。

## 对 wiki 的映射

- [Click-and-Traverse](../../wiki/entities/cn-os-click-and-traverse.md)
