# zhihu_jagger_task_space_fb_bfm_intact_mimic_vla

> 来源归档（blog / 知乎专栏）

- **标题：** 数据是我们唯一要关心的事情吗？对 FB，BFM-zero, INTACT, Mimic，VLA 在任务空间中不同表征的思考
- **类型：** blog
- **作者：** Jagger（知乎 @nike-20-31；专栏「Have Technical Fun.运动控制、规划」）
- **原始链接：** https://zhuanlan.zhihu.com/p/2066468645300180732
- **发布 / 编辑：** 2026-08-02 发布；2026-08-04 编辑（上海）
- **入库日期：** 2026-08-04
- **一句话说明：** 以 RoboParty Lab 近期 MimicLite / UFO(BFM-Zero) / INTACT 为锚，把 FB、BFM-Zero、INTACT、Mimic、VLA 统一到「任务空间 latent 几何」坐标系——主张数据之外，目标函数定义域（欧氏曲线 / Goal-Reach 子空间 / 正交任务球）决定覆盖形状、可拼接性与 OOD 转移能力；并附 RL 相对 MPC 更快解决人形行走的接触平滑读法。
- **沉淀到 wiki：** [`wiki/comparisons/fb-bfm-zero-intact-mimic-vla-task-space.md`](../../wiki/comparisons/fb-bfm-zero-intact-mimic-vla-task-space.md)

## 核心摘录（归纳，非全文）

### 1) 总命题：数据不是唯一变量

- Data-based 路线假设：数据若能铺满任务空间，就可少做人为任务定义。
- 反诘：即便数据「看起来够大」，模型如何利用数据仍取决于 **目标函数为何而设、在什么空间上定义**；Full cover 难，往往难在「目标 + 空间」层，而不只是样本不够。

### 2) FB 理论：完整球面几何的任务空间

- 任务空间无穷维；Forward–Backward（FB）假设可用 **D 维正交 latent \(Z\)** 线性组合近似覆盖 goal reaching / motion tracking / reward guide 等方向。
- 归一化后呈球面几何：不同方向向量 ≈ 不同任务；BFM-Zero 取 \(D=256\)（最重要的奇异方向）。

### 3) BFM-Zero：数据集约束下的拟人子任务空间

- 论文锚点：arXiv:2511.04131；工程侧常与 RoboParty UFO 同读。
- 三个关键处理：
  1. **风格子流形**：LAFAN 等 mocap → retarget → Discriminator，把任务球限制到拟人子区域；
  2. **防坍缩正则**：大权重保持 \(D\) 维正交/不相关（文称此项 loss 权重最大）；
  3. **off-policy \(Z\) 采样**：\(Z\) 可来自轨迹嵌入，也可从 \([-1,1]\) 均匀随机采样 → 未见完整 A→B 转移数据时仍可能组合出 A→B。
- 价值读法：奖励的是「当前意图 \(z\) 下未来能稳定收敛」的方向，而非瞬时欧氏误差最快下降。

### 4) INTACT：Goal-Reach specific 子空间

- 论文锚点：arXiv:2607.26056（JEPA WM；意图→动作无搜索）。
- 不铺满奖励方向球面，而把 latent 收成 **Goal Reach 可用子区域**；\(Z\) 采样完全来自数据集。
- 与 FB/Mimic 对照：
  - 多任务 transition 通常比 FB 更依赖「见过 A→B 路径」；
  - 相对 Mimic：见过一条倒地爬起后更易泛化到任意倒地姿态（意图空间 vs 欧氏轨迹）。

### 5) Mimic：欧氏 Motion Tracking 在任务球上的曲线投影

- SONIC / MimicLite 等：以欧氏跟踪误差为奖励，PPO 沿误差下降最快方向学 **欧氏曲线**；投影到 \(Z\) 球上仍是曲线网，曲线间距过大即 OOD、难转移。
- Termination：关则易被 OOD 样本污染梯度（倒地时「乱动」可比起身更快减小瞬时跟踪误差）；开则靠阈值保证奖励梯度不消失。
- 精度优势：梯度直接指向下一时刻轨迹误差减小；相对 FB 有时滞与 \(D\) 维近似精度代价。
- FSQ/VQ-VAE：离散 codebook + 正则可扩大任务隐空间投影面积、缓解坍缩（SONIC）。

### 6) VLA：语言–视觉指令的稀疏语义投影

- 预训练 VLM（CE/NLL）决定语义支撑在 \(Z\) 上铺多开；后训练 action head（离散 CE 或 diffusion/flow MSE）主要改善「每个语义点怎么动」，**不自动扩大任务空间覆盖**。
- OOD 类比 Mimic：换说法 / 换背景常失效，类似曲线间距过大。
- World Model 叙事优势：更直接面对任务空间表征；但多数仍停在重建/短程预测，尚未形成 FB 式可线性组合任务坐标，也未必具备 BFM-Zero 式防坍缩与 off-policy 拼接。

### 7) 旁支：为何 RL 比 MPC 更快解决人形行走

- 人形相对机械臂/车/无人机：高维非线性、**动力学突变（多接触）**、低静态稳定裕度、浮动基——底层既要 Task 又要 Whole-Body Dynamic。
- RL：离线多样本期望回报把接触 0/1 突变平均成接触概率×力，梯度被采样分布平滑（代价是 sim2real gap）。
- MPC：须显式做接触检测 / 规划 / 约束（软接触 PD 或 LCP 互补）；实时精确梯度难。
- RL 局限：折现视野在 50 Hz、\(\gamma\approx0.98\)–\(0.99\) 时约 1–2 s；对稀疏奖励与极高频短窗信息不友好。

### 8) 收束三问（文末）

1. Task 坐标是否有意义（可组合 / 拼接 / 提示，而非只会重建或短程预测）；
2. Dynamic 是否进表征或进闭环（接触、力、阻抗、浮动基）；
3. 从底层到算法，每一步如何决定系统带宽与频域取舍。

## 对 wiki 的映射

- [fb-bfm-zero-intact-mimic-vla-task-space](../../wiki/comparisons/fb-bfm-zero-intact-mimic-vla-task-space.md)（本次升格主页面）
- [paper-bfm-zero](../../wiki/entities/paper-bfm-zero.md)
- [paper-intact](../../wiki/entities/paper-intact.md)
- [mimiclite](../../wiki/entities/mimiclite.md)
- [roboparty-ufo](../../wiki/entities/roboparty-ufo.md)
- [sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md)
- [vla](../../wiki/methods/vla.md)
- [behavior-foundation-model](../../wiki/concepts/behavior-foundation-model.md)
- [roboparty-lab-party-os-technology-map](../../wiki/overview/roboparty-lab-party-os-technology-map.md)
- [mpc-vs-rl](../../wiki/comparisons/mpc-vs-rl.md)
- [bfm-category-01-forward-backward-representation](../../wiki/overview/bfm-category-01-forward-backward-representation.md)

## 相关外部锚点

| 锚点 | URL |
|------|-----|
| 本文 | https://zhuanlan.zhihu.com/p/2066468645300180732 |
| 作者主页 | https://www.zhihu.com/people/nike-20-31 |
| BFM-Zero | https://arxiv.org/abs/2511.04131 |
| INTACT | https://arxiv.org/abs/2607.26056 |
| FB / BFM 论文索引 | https://github.com/friedrichyuan/awesome-bfm-papers |
| RoboParty Lab | https://lab.roboparty.com/ |
| 作者前作 Know-How（文内互链） | https://zhuanlan.zhihu.com/p/1993986785630499920 |

## 可信度与使用边界

- 本文为**个人技术洞察 / 策展对比**，非同行评议综述；对 FB / BFM-Zero / INTACT / Mimic / VLA 的「球面几何」读法是作者统一坐标系下的解释框架，工程选型应以各论文实验与开源实现为准。
- 文中「\(D=256\)」「LAFAN 40 条」「最大 loss 权重」等细节转述自公开论文与作者理解，复核以 arXiv PDF 为准。
- 抓取说明：知乎专栏对 Jina Reader 返回 403；入库正文经 Camoufox（agent-reach 工具链）渲染后摘录。

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] 初步 wiki 页面映射确认
- [x] 升格对比页
