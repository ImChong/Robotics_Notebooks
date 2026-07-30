# intact_arxiv_2607_26056

> 来源归档（ingest）

- **标题：** INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models
- **类型：** paper
- **来源：** arXiv:2607.26056（2026-07-28）
- **作者：** Junhan Sun, Hao Zhao, Guofeng Zhang
- **机构：** 浙江大学（ZJU）CAD&CG；清华大学 AIR；InSpatio；RoboParty Lab
- **入库日期：** 2026-07-30
- **最后更新：** 2026-07-30
- **项目页：** <https://zju3dv.github.io/INTACT-JEPA/>
- **代码（规范仓）：** <https://github.com/zju3dv/INTACT-JEPA>（MIT；**代码/权重 Coming Soon**，文档与结果审计已上）
- **代码（RoboParty 镜像）：** <https://github.com/Roboparty/INTACT-JEPA>（fork；同内容）
- **Lab：** <https://lab.roboparty.com/>
- **一句话说明：** 端到端 JEPA：共享四槽语法把物理意图与部署意图映射为动作律，使条件均值成为无搜索策略；LeWM 四任务 Direct 2.9–5.5 ms，宏成功率约 95%（相对宽搜索 CEM 延迟约 **300×**）。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / Teaser）

- **链接：** <https://arxiv.org/abs/2607.26056>
- **核心贡献：** 前向 latent 世界模型回答「动作会怎样改变场景」，但目标条件控制常靠测试时 **CEM/MPPI 搜索** 反解动作。INTACT（INtent-To-ACTion）是端到端 JEPA：转移提供物理意图 \(z_{t+1}-z_t\)，未来目标提供部署意图 \(\operatorname{sg}(z_g)-z_t\)；局部/目标调用共享 **四槽语法** 与参数；通过同一预测器诱导的 **action-law 语义** 对齐意图族（非点对点 latent 相等）。不对称端点梯度接地物理后继、锚定未来目标。条件均值可作 **search-free policy**；采样保留多样性或可选局部验证。官方 LeWM 四任务单 epoch 零搜索达 **85.78% / 100% / 97.67% / 97.89%**；可选局部 CEM（384 vs 9000 候选，**23.44×**）宏成功率 **96.86%**；共享四任务编码器 E5 Direct 宏 **89.39%**；Direct 推理 **2.9–5.5 ms**。
- **对 wiki 的映射：**
  - [INTACT 论文实体](../../wiki/entities/paper-intact.md)
  - [DWM Separating World Effects](../../wiki/entities/paper-dwm-separating-world-effects.md)（LeWM 族对照）
  - [V-JEPA 2](../../wiki/entities/paper-vjepa2.md)

### 2) 同构意图→动作接口（§3）

- **输入语法：** \(x_t(m_t)=[z_t,\,m_t,\,z_t\odot m_t,\,A(a_{t-1})]\)，共享 \(G_\eta\) 输出 \(p_\eta(a_t\mid x_t)\)。
- **两意图实例：** \(m^{\mathrm{local}}=z_{t+1}-z_t\)（梯度附着）；\(m^{\mathrm{goal}}=\mathrm{sg}(z_g)-z_t\)（目标停梯度）。
- **损失：** 局部 + 目标两路 NLL（相对同一示范动作），**无**端点间直接匹配损失；前向 JEPA 保留可滚出信息。
- **控制：** Direct = 条件均值（无搜索）；Guarded A = 以 Direct 为中心的小预算局部 CEM。
- **对 wiki 的映射：**
  - [生成式世界模型](../../wiki/methods/generative-world-models.md)
  - [模型基强化学习](../../wiki/methods/model-based-rl.md)

### 3) 评测：单任务与共享编码器（§4–5）

- **协议：** 官方 LeWM 四任务；单任务一 epoch；共享编码器多任务。
- **单任务 Direct 宏约 95.33%**；Guarded 96.86%。相对匹配起点 LeWM CEM，Cube 审计约 **+31.7 pp**（98.7% vs 67.0%）。
- **诊断：** predicted–expert action-family kNN 与 Direct SR 相关 \(r\approx 0.95\)–\(0.98\)。
- **对 wiki 的映射：**
  - [INTACT 论文实体](../../wiki/entities/paper-intact.md)
  - [物理保真输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md)

### 4) 局限与开源（§6 / 步骤 2.5）

- **局限：** 三 seed 粗糙；动作商仅在示范支撑上可辨；Direct 高斯可能掩盖多模态；gauge 等价依赖任务流形。
- **开源核查（2026-07-30）：** 项目页 <https://zju3dv.github.io/INTACT-JEPA/> + 规范仓 `zju3dv/INTACT-JEPA`（MIT）已上线方法/结果/复现文档；`docs/RELEASE.md` 标明训练代码与 checkpoint **Coming Soon / Stage 0–1** → **部分开源（文档与仓库骨架；可运行训练/推理入口待发）**。RoboParty 组织另有 fork 镜像 `Roboparty/INTACT-JEPA`（同 Stage，便于 Lab 导航）。
- **与 LeWM 的分工（宣传/README_CN 口径）：** LeWM 类前向 latent WM 学「动作会产生什么效果」；INTACT 补「为了实现意图应执行什么动作」的同构读出，使部署不必再宽搜动作序列。

## 关键数字速查

| 指标 | 数值 |
|------|------|
| Direct 四任务 SR | **85.78 / 100 / 97.67 / 97.89%** |
| Direct 宏（一 epoch） | **~95.33%** |
| Guarded 宏 | **96.86%**（384 候选） |
| 共享 E5 Direct 宏 | **89.39%** |
| Direct 延迟 | **2.9–5.5 ms** |
| vs 全量 CEM 采样 | **23.44×** 更少（Guarded） |

## 其他公开资料

- **项目页：** [sites/intact-jepa-github-io.md](../sites/intact-jepa-github-io.md)
- **规范仓：** [repos/intact-jepa.md](../repos/intact-jepa.md)
- **RoboParty 镜像：** [repos/roboparty-intact-jepa.md](../repos/roboparty-intact-jepa.md)
- **Lab：** [sites/lab_roboparty_com.md](../sites/lab_roboparty_com.md)
- **arXiv HTML：** <https://arxiv.org/html/2607.26056>

## 当前提炼状态

- [x] sources 归档
- [x] 升格 wiki 实体页
- [x] 交叉 LeWM 族 / JEPA / 世界模型总览
