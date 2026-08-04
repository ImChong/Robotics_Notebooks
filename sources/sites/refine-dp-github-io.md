# refine-dp.github.io/REFINE-DP（项目页）

> 来源归档（ingest）

- **标题：** REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via Reinforcement Learning
- **类型：** site / project-page
- **官方入口：** <https://refine-dp.github.io/REFINE-DP/>
- **入库日期：** 2026-08-02
- **一句话说明：** Georgia Tech IRIM 配套站点：三阶段管线示意、Booster T1 真机四任务视频、联合优化与数据效率曲线；Code 按钮为无链接占位。

## 页面公开信息（检索自 2026-08-02）

| 资源 | URL / 状态 |
|------|------------|
| 项目首页 | <https://refine-dp.github.io/REFINE-DP/> |
| arXiv | <https://arxiv.org/abs/2603.13707> |
| arXiv HTML | <https://arxiv.org/html/2603.13707> |
| 机构 PDF 镜像 | <https://lab-idar.gatech.edu/wp-content/uploads/2026/03/RAL_humanoid_diffusion_2026.pdf> |
| GitHub Pages 仓 | <https://github.com/REFINE-DP/REFINE-DP>（`gh-pages`：仅 `index.html` + `assets/`） |
| **代码** | **未开源** — 页眉 Code 为无 `href` 的 `<span>`；仓库无训练/推理入口 |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **三阶段：** (a) VR/启发式示教经冻结低层 RL 控制器采数；(b) DiT 骨干 DP 预训练输出基座速度 + 双手 SE(3)；(c) PPO/DPPO 联合微调 DP 与低层 loco-manip 控制器。
2. **数据效率：** 约 50 条遥操作轨迹微调可达约 90%+ 仿真成功率，纯扩数据约需 1000 条才到同等量级（约 20×）。
3. **真机：** Booster T1（29 DoF）；任务含走拾、长程 pick-place、开门穿越、上台取物。
4. **联合优化：** 单独调低层可把长程 pick-place SR +18%；联合优化把达 90% SR 的迭代从约 40 降到约 20，并改善跟踪与平滑。

## 对 wiki 的映射

- [`wiki/entities/paper-loco-manip-161-157-refine-dp.md`](../../wiki/entities/paper-loco-manip-161-157-refine-dp.md)
- [`sources/papers/refine_dp_arxiv_2603_13707.md`](../papers/refine_dp_arxiv_2603_13707.md)
