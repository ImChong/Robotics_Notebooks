# fa-rdp.github.io（项目页）

> 来源归档（ingest）

- **标题：** FA-RDP: A Frequency-Adaptive Reactive Diffusion Policy for Contact-Rich Manipulation
- **类型：** site / project-page
- **官方入口：** <https://fa-rdp.github.io/>
- **入库日期：** 2026-08-02
- **一句话说明：** SJTU / 创智 / Noematrix 配套站点：三任务真机对比视频、多模态与蒸馏消融、Table I 成功表；Code 为 coming soon。

## 页面公开信息（检索自 2026-08-02）

| 资源 | URL / 状态 |
|------|------------|
| 项目首页 | <https://fa-rdp.github.io/> |
| Paper（站点 PDF） | <https://fa-rdp.github.io/root.pdf> |
| arXiv | <https://arxiv.org/abs/2607.28596> |
| GitHub Pages 仓 | <https://github.com/zhuolifeng/FA-RDP>（仅站点 + `releases/v1.0` 视频） |
| **代码** | **未开源（coming soon）** — Code 按钮 `href="#"`；仓库无训练/推理入口 |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **相位切换：** 接触前多模态低频率多步采样；接触后高频一步蒸馏采样。
2. **MCD：** 在动作流形上蒸馏，保留 DDPM 残差监督。
3. **结果：** 三任务平均成功率 **81.7%**，高于 DP / RDP / ImplicitRDP / 力回归基线。
4. **视频：** Box / Switch / Button 成功与基线失败对照（托管在 GitHub Releases）。

## 对 wiki 的映射

- [`wiki/entities/paper-fa-rdp.md`](../../wiki/entities/paper-fa-rdp.md)
- [`sources/papers/fa_rdp_arxiv_2607_28596.md`](../papers/fa_rdp_arxiv_2607_28596.md)
