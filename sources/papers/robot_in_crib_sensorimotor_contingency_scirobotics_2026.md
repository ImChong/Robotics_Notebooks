# Robot in a crib: How a playing robot helps us understand sensorimotor contingency learning（Science Robotics, 2026）

> 来源归档（ingest）

- **标题：** Robot in a crib: How a playing robot helps us understand sensorimotor contingency learning
- **类型：** paper / developmental robotics / sensorimotor contingency / mobile paradigm / iCub
- **期刊：** Science Robotics, 2026-08-26（Vol. 11, Issue 117）
- **DOI：** <https://doi.org/10.1126/scirobotics.aed4106>
- **作者：** Josua Spisak†、Sergiu Tcaci Popescu†、Lukas Rustler、Stefan Wermter、J. Kevin O'Regan、Matej Hoffmann‡（† 共同一作；‡ 通讯作者之一）
- **机构：** 汉堡大学（University of Hamburg，Knowledge Technology Group）；捷克理工大学电气工程学院（CTU FEE，Cybernetics）；巴黎西岱大学（Université Paris Cité）与 CNRS；等
- **平台：** iCub 人形机器人 + 木制婴儿床 + 悬挂移动玩具架（mobile）；头部摄像头视觉反馈；弹性绳连接单臂
- **全文获取：** **无公开 arXiv 预印本**（截至 2026-09-01）；正式版见 Science Robotics DOI。同团队仿真建模工作见 [arXiv:2504.17939](https://arxiv.org/abs/2504.17939)（已发表于 IEEE TCDS 2026，非本文逐字预印本）。
- **媒体与新闻：** [CTU FEE 新闻稿](https://fel.cvut.cz/en/what-s-on/news/84994-how-does-a-child-learn-that-it-can-change-the-world-around-it-a-humanoid-robot-offers-new-answers)；[照片/视频 Google Drive](https://drive.google.com/drive/folders/1ZbdNLyJZWgOXK1fqgMFGd1VkknGU69H5?usp=sharing)
- **代码与数据：** **部分开源**（截至 2026-09-01）：同课题组的 **仿真计算模型** 已开源 [`ctu-vras/mobile-paradigm-model`](https://github.com/ctu-vras/mobile-paradigm-model)；**真机 iCub 摇篮实验** 的完整部署/控制栈在 CTU 新闻与 Google Drive 中**未列 GitHub**。
- **入库日期：** 2026-09-01
- **一句话说明：** 用躺在婴儿床里的 iCub 复现经典 mobile paradigm，以预测损失 + 好奇心损失驱动「人工脑」，证明感觉运动偶联学习不止「连接肢动得更多」，而是涌现至少四种利用策略，并与婴儿 2–4 月龄探索–利用转变对应。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.aed4106](https://doi.org/10.1126/scirobotics.aed4106) | Science Robotics 正式版 |
| 机构新闻 | [CTU FEE 2026-08-31](https://fel.cvut.cz/en/what-s-on/news/84994-how-does-a-child-learn-that-it-can-change-the-world-around-it-a-humanoid-robot-offers-new-answers) | 实验动机、作者引述、媒体包 |
| 早期新闻 | [CTU FEE 预研报道](https://fel.cvut.cz/en/what-s-on/news/81170-what-goes-on-in-a-child-s-mind-a-humanoid-robot-in-a-crib-is-helping-scientists-from-the-fee-ctu-in-prague-to-understand-early-cognitive-development) | 提及 arXiv 仿真预印本与真机进展 |
| 仿真姊妹篇 | [arXiv:2504.17939](https://arxiv.org/abs/2504.17939) | 同作者组 mobile paradigm 计算模型（IEEE TCDS 2026） |
| 仿真代码 | [mobile-paradigm-model](https://github.com/ctu-vras/mobile-paradigm-model) | PyTorch 仿真；非真机 YARP 栈 |
| 媒体 | [Google Drive 相册/视频](https://drive.google.com/drive/folders/1ZbdNLyJZWgOXK1fqgMFGd1VkknGU69H5?usp=sharing) | 实验装置与 Science Robotics 截图 |
| iCub 平台 | [`wiki/entities/paper-ergocub-shared-embodied-intelligence.md`](../../wiki/entities/paper-ergocub-shared-embodied-intelligence.md) | iCub 系人形研究脉络 |

## 摘要级要点

- **问题：** 发展心理学经典 **mobile paradigm**（Rovee-Collier）用「连接肢蹬动 → 玩具架动」研究婴儿如何发现 **感觉运动偶联（sensorimotor contingency）**；长期默认指标是「连接肢运动量增加」，但婴儿实验结果存在变异与复现不一致。
- **方法：** 将婴儿替换为 **iCub**，躺在特制木床中，单臂经弹性绳连悬挂 mobile；头部摄像头感知玩具运动；**神经网络「人工脑」** 不预设哪只手臂被连接，由 **预测损失**（动作/环境）与 **好奇心损失**（意外后果）联合驱动关节指令；真机 + Gazebo 仿真共 **七种条件**。
- **偶联检测（外部观察者）：** 用 MLP 对双臂轨迹分类「哪只手臂被连接」；准确率显著高于 50% 即认为出现系统性偶联相关运动模式（非预设「动得更多」）。
- **核心发现 1：** 七种条件下 **连接/非连接臂总路径长度无实质差异**——挑战「运动量增加 = 学到偶联」的单指标解读。
- **核心发现 2：** 复杂条件下决策树识别 **四种主导运动策略**（约覆盖 87% 样本）：高效小幅度精准利用 vs 大范围探索等；连接臂有时 **动得更少** 但运动 **更精准有效**。
- **核心发现 3：** 内部决策树显示 **有效运动策略** 对应低预测/好奇心损失（利用主导）；**探索运动策略** 对应高运动预测损失（探索主导）——与婴儿 2 月龄（全身乱动）vs 4 月龄（选择性动连接肢）的 **探索–利用权衡** 假说一致。
- **实验设计亮点：** **重放条件** 回放相同视觉序列但断开动作–感官因果，隔离「偶联存在」变量；复杂真机条件 MLP 准确率约 **65%**，简单真机仅 **53%**（不显著）——支持「甜蜜点」难度假说。

## 核心摘录（面向 wiki 编译）

### 1) 七种实验条件

| 环境 | 条件 | 要点 |
|------|------|------|
| 真机 | 复杂 | 彩色多部件玩具架，运动非确定、低可预测 |
| 真机 | 简单 | 泡沫管 + 滑轮，高确定性连接 |
| 真机 | 对照 | 无 mobile，无偶联 |
| 仿真 | 复杂 / 简单 | Gazebo 重建；简单条件视觉反馈≈手臂运动 |
| 仿真 | 重放复杂 / 简单 | 回放先前视觉，但动作不再影响玩具 |

- 复杂真机 **20** 次重复；仿真各 **60** 次。

### 2) MLP 偶联检测准确率（论文报道量级）

| 条件 | 分类准确率 | 显著？ |
|------|-----------|--------|
| 真机复杂 | 65±4% | 是 |
| 真机简单 | 53±4% | 否 |
| 真机对照 | 48±7% | 否 |
| 仿真复杂 | 67±3% | 是 |
| 仿真简单 | 58±3% | 是 |
| 重放仿真复杂/简单 | ~45–48% | 否 |

### 3) 四种运动策略（复杂条件，决策树）

1. **有效运动–右臂连接（~25%）：** 非连接左臂窄域静止，连接右臂小幅精准运动。
2. **有效运动–左臂连接（~75%）：** 连接左臂在绳绷紧区域小幅运动。
3. **探索运动–右臂连接（~25%）：** 左臂大范围探索，右臂局部小动。
4. **探索运动–左臂连接（~10%）：** 连接左臂大幅度探索，右臂贴躯干。

### 4) 与仿真姊妹篇关系

- [arXiv:2504.17939](https://arxiv.org/abs/2504.17939) 在 **简化虚拟肢体** 上复现经典「连接肢偏好」与 **extinction burst**；本文将同一认知机制 **具身到真机 iCub**，并揭示 **多策略行为光谱** 与单指标失效。
- 仿真代码 [`mobile-paradigm-model`](https://github.com/ctu-vras/mobile-paradigm-model) 可调 fatigue/curiosity/prediction 权重；**真机闭环代码未公开**。

### 5) 开源状态（步骤 2.5 核查）

- **已开源：** 仿真计算模型 `ctu-vras/mobile-paradigm-model`（PyTorch，`python main.py`）。
- **未开源 / 未列链：** iCub 真机 YARP 控制、七种条件实验脚本、MLP 观察者训练包；CTU Google Drive 仅照片/视频。
- **对 wiki 的映射：** 实体页「工程实践」区分仿真可复现 vs 真机装置；源码时序图仅覆盖 **仿真模型** 运行时。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-robot-in-crib-sensorimotor-contingency.md`](../../wiki/entities/paper-robot-in-crib-sensorimotor-contingency.md)**
- 交叉：**[`wiki/entities/paper-ergocub-shared-embodied-intelligence.md`](../../wiki/entities/paper-ergocub-shared-embodied-intelligence.md)**（iCub 平台）、**[`wiki/entities/paper-notebook-learning-with-pycub-a-simulation-and-exercise-fr.md`](../../wiki/entities/paper-notebook-learning-with-pycub-a-simulation-and-exercise-fr.md)**（iCub 仿真教学栈）
- 站点归档：**[`sources/sites/ctu-fee-robot-in-crib-scirobotics-2026.md`](../sites/ctu-fee-robot-in-crib-scirobotics-2026.md)**
- 代码归档：**[`sources/repos/ctu-vras-mobile-paradigm-model.md`](../repos/ctu-vras-mobile-paradigm-model.md)**
