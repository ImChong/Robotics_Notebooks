# SCAR: Self-Supervised Continuous Action Representation Learning（Aether AI 博客）

> 来源归档（blog / Aether AI 官方 Field notes #08）

- **标题：** SCAR: Self-Supervised Continuous Action Representation Learning
- **类型：** blog（论文解读页，配套 arXiv 论文；页面版式为独立的「Robot Learning Notes」模板）
- **作者 / 组织：** 页内参考文献署名 Liu, H., Feng, F., Fu, M., Wang, X., Lu, H., & Huang, B. / 托管于 Aether AI 博客（aetherlabs.ai）
- **原始链接：** <https://aetherlabs.ai/articles/scar-self-supervised-continuous-action-representation-learning.html>
- **博客索引：** <https://aetherlabs.ai/blog.html>（编号 08 · SCAR；标签 World Models · Latent Actions）
- **发表日期：** 2026-08-09（博客发布日；arXiv v1 早于此，为 2026-05-13）
- **入库日期：** 2026-10-10
- **抓取方式：** `curl` 抓取静态 HTML；页面主要是理论要点 + 方法图 + 对比视频，**无数值表**；数值取自 arXiv HTML（见论文归档）
- **一句话说明：** 把「动作」当作视觉变化中的一个独立因子：逆动力学模型（IDM）从潜观测对推断随机潜动作，前向动力学模型（FDM）在预训练视频骨干上以潜动作为条件预测未来；KL 把后验拉向标准高斯以限制外观捷径，梯度反转（GRL）对抗本体分类器以去除本体信息，得到可跨本体迁移的统一潜动作接口。

## 开源 / 项目页核查（步骤 2.5，截至 2026-10-10）

| 项 | 结论 |
|----|------|
| 论文 | arXiv:2605.16412（见 [论文归档](../papers/scar_arxiv_2605_16412.md)） |
| 项目页 | 无独立项目页；博客页顶栏「Project」只是页内锚点 `#results` |
| 代码 / 权重 | **未列出**：博客与 arXiv 摘要均无 GitHub / Hugging Face 链接 |

## 核心摘录（归纳，非全文）

### 主张

- 具身世界模型通常以原始动作命令为条件，但命令绑定于具体本体、控制器与标定。「原始机器人命令是一个局部接口：同一条命令在不同身体、控制器、场景下意味着不同的物理干预。」
- SCAR 把本体当作 **干扰变量**：潜动作要对预测有用，但不应泄露是哪台机器人产生的。

### 理论要点（充分条件，理想化假设）

- 生成过程：本体无关潜动作 \(u_t\) → 本体相关命令 \(a^e_t=h(u_t,e)\) → 转移 \(s_{t+1}=F(s_t,a^e_t)\) → 渲染 \(x_t=R(s_t)\)。
- 渲染与动力学在数据支撑上单射时，全局最优的 IDM 可把实际动作恢复到连续可逆重参数化 \(\tilde a_t=\rho(a^e_t)\)。
- 最优本体分类器下 \(\min_\omega\mathcal L_{CE}=H(e\mid z)=H(e)-I(e;z)\)，梯度反转把编码器推向 \(z\perp e\)；线性证明设定中本体簇中心张成干扰子空间 \(V\)，不变性迫使 \(\mathrm{row}(M)=V^\perp\)。
- 结论：统一潜动作空间可恢复到 **每个本体一个可逆双射** \(z=\Phi_e(u)\)；从 \(z\) 还原 \(u\) 仍需本体 ID。
- 作者自己的限定：这是理想化结构化干扰 + 非塌缩预测假设下的 **充分条件分析**，不是说任何训好的模型都能完美零样本迁移；前向预测防止平凡不变塌缩，KL 限制无约束视觉编码。

### 方法四步

1. IDM 读潜观测对，输出随机潜动作 token；
2. FDM 用潜动作条件化预训练视频动力学骨干；
3. KL 限制容量 + GRL 去本体信息；
4. 推理时，上下文条件的 **action-to-latent 控制器**（A2L）把原始命令序列映射进潜动作空间，恢复可控性。

### 演示（定性视频）

- 跨本体动作迁移：从源本体轨迹抽潜动作，施加到目标本体视觉上下文；对比 raw latent / +KL / +KL&GRL。
- 少样本目标本体世界模型：Raw GT action / Raw latent / SCAR（KL&GRL）/ Controller recovery。

## 可信度边界

- 博客页本身无定量结果；本页数值全部来自 arXiv HTML。
- 实验只在 Procgen（虚拟本体）与 RoboTwin 单个任务 `place_a2b_left`（迁移到 `place_a2b_right`）上做，均为生成质量指标（SSIM / PSNR / MSE），**没有闭环策略成功率或真机实验**；论文结论段也把真实机器人数据、野外视频与闭环策略学习列为后续工作。

**对 wiki 的映射：** [paper-scar-continuous-action](../../wiki/entities/paper-scar-continuous-action.md)
