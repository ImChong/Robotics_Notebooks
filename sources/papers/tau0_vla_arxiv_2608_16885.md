# τ₀-VLA（分层机器人基础模型 + 世界模型引导测试时计算）

> 来源归档（ingest）

- **标题：** τ₀-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.16885>
  - <https://tau0-vla.github.io/tau0-vla.pdf>
  - <https://tau0-vla.github.io/>
- **代码：** <https://github.com/sii-research/tau-0-vla>
- **权重：** <https://huggingface.co/sii-research/tau-0-vla>
- **机构：** 上海创智学院；智元机器人 Finch 团队；香港中文大学
- **发布日期：** 2026-07-27（项目页 / GitHub News）
- **入库日期：** 2026-08-19
- **一句话说明：** **分层 VLA 基础模型**：记忆增强 **高层子任务策略** 在不确定时用 **世界模型引导 beam search（TTC）** 比较候选子任务的 **想象后果** 再提交；**通用低层 VLA**（Qwen3.5 + MoT flow 专家、**40 维** 统一动作）在 **40,115 h** 异构真机数据上预训练，支撑 **13–25 步** 长程真机任务与跨本体部署。

## 核心摘录（MVP）

### 1) 子任务级测试时计算：propose–predict–evaluate

- **摘录要点：** 多数分层 VLA 的高层决策仍是 **单次前向**；τ₀-VLA 把 **下一子任务生成** 写成 **可扩展算力** 的推理问题。高层策略 \(\mu\) 含 **Proposal P、World Model W、Value V、Reflective F**；token 置信度路由：高置信 **快路径** 直接采纳 \(z^{\mathrm{dir}}_t\)，否则触发 **TTC**。TTC 对候选子任务序列做 **beam search**：P 提出开放候选、W 预测 **单 head 相机终端图像**、V 对 **(ℓ, z, ô)** 打分；保留束交给 F 生成最终子任务（可超出候选集）。
- **对 wiki 的映射：**
  - [τ₀-VLA](../../wiki/entities/paper-tau0-vla.md) — 高层 TTC 与四模型接口。
  - [VLA](../../wiki/methods/vla.md) — 分层 vs 整任务指令直出动作的对照。

### 2) 可修订执行记忆（execution memory）

- **摘录要点：** 高层上下文 \(h_t = (\ell, M_{t-1}, z^\star_{t-1}, o_t)\)；P 用当前观测更新 **执行记忆** \(M_t\)。训练时对演示派生记忆 **扰动**（滞后、超前、错标），教策略 **前进 / 回滚 / 重试** 修复进度记录，**无需额外纠错数据集**；论文报告 **next-subtask 准确率 +11.0 pt**。
- **对 wiki 的映射：**
  - [τ₀-VLA](../../wiki/entities/paper-tau0-vla.md) — 记忆机制与分布偏移鲁棒性。

### 3) 通用低层 VLA 与 40 维统一动作

- **摘录要点：** 低层 \(\pi_\theta\) 用 **预训练 VLM 骨干 + Mixture-of-Transformers action expert**，**条件 flow matching** 输出 **H 步 action chunk**；**40 维** 状态/动作覆盖 **EEF、臂关节、夹爪、腰、移动底盘**，各 embodiment 映射并 **mask 未用槽位**。整任务直出时 \(c_t=\ell\)；分层时 \(c_t=z^\star_t\)。**40,115 h** 异构真机 + 多模态共训建立 **跨本体 generalist** 执行接口。
- **对 wiki 的映射：**
  - [τ₀-VLA](../../wiki/entities/paper-tau0-vla.md) — 低层结构与后训练栈。
  - [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 异构小时与统一动作空间对照。

### 4) 长程真机与分层增益

- **摘录要点：** 四类 **13–25 有序步** 任务（Clean Room、Prepare Ingredients、Stir Fry、Make Milk Tea），episode 最长 **12 分钟**；含导航、搜寻、关节物体、工具、烹饪与失败恢复。同低层策略下：**τ₀-VLA 直出整任务 27.5%** 平均成功率 vs **分层 Plan Once 45.0%**（10 次/任务）。对照 **π₀.₅ 22.5%**、**GR00T N1.7 2.5%**、**LingBot-VLA 0%**（项目页表）。TTC 在分布偏移 **Book Organization** 上 next-subtask **74.0%** vs Plan Once **50.0%**；固定低层时 Milk Tea 成功 **5/10→7/10**、Clean Room **6/10→9/10**。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md) — 长程 household 操纵语境。
  - [τ₀-WM](../../wiki/entities/tau0-world-model.md) — 同生态 **测试时想象** 对照（WM 在动作 chunk 级 vs VLA 在子任务级）。

### 5) 开源状态（截至 2026-08-19）

- **摘录要点：** **已开源（部分）**：低层 **HF 权重**、**LeRobot v3** 后训练范例、`deploy.server` 开环评测、**example AgiBot World** 子集（Apache-2.0）。README **[2026.08.19]**：**高层 policy 组件将逐步发布**；当前公开 serving 仅 **joint-control checkpoint**（原生 EEF 训练可用，**EEF serving 本版不支持**）。
- **对 wiki 的映射：**
  - [sii-research/tau-0-vla](../../sources/repos/sii_research_tau_0_vla.md) — 仓库布局与部署契约。

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-tau0-vla.md` 新建，并与 τ₀-WM / π₀.₅ / VLA / manipulation 交叉引用
- [ ] 高层 TTC 代码与权重随官方逐步发布后补 `sources/repos/` 与实体页「开源状态」
