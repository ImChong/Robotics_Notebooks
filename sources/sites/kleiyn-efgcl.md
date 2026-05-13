# EFGCL（KLEIYN 项目页 / RA-L 2026）

- **类型**：研究项目页 + 期刊论文（原始摘录）
- **收录日期**：2026-05-13
- **项目页**：<https://keitayoneda.github.io/kleiyn-efgcl/>
- **正式出版**：IEEE Robotics and Automation Letters, 2026, Vol. 11, No. 5, pp. 5907–5913；DOI：<https://doi.org/10.1109/LRA.2026.3675955>

## 一句话

**EFGCL（External Force-Guided Curriculum Learning）**：在仿真训练中对腿足机器人施加**可随成功率衰减的外部辅助力**，类比体操中的 **spotting（保护/托举）**，使智能体在早期就能**物理上经历完整成功轨迹**，从而在**简单稀疏奖励**下学会跳跃、后空翻、侧空翻等高动态全身动作，并报告 sim2real 到四足平台 **KLEIYN**。

## 为什么值得保留

- 把「**物理引导探索**」表述为与**奖励塑形 / 参考轨迹**正交的第三条路：辅助力是**训练期环境修改**，不进入部署策略的观测或控制律。
- 与工业界已出现的「**虚拟辅助力/扳手 + 课程衰减**」叙事（如 ZEST）形成对照，但本工作给出**四足实机**与**价值函数收敛加速**的量化叙述（项目页声称约 200 iter vs 基线 1000+ iter 量级对齐价值分布；Jump 任务约 **2×** 样本效率）。

## 项目页摘录（一级叙事）

来源：<https://keitayoneda.github.io/kleiyn-efgcl/>（2026-05-13 抓取）

- **标题**：*EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force-Guided Curriculum Learning*
- **作者（页面）**：Keita Yoneda, Kento Kawaharazuka, Kei Okada；单位：**JSK Lab, The University of Tokyo**
- **摘要口径**：
  - 腿足机器人学**高动态全身运动**时，失败风险高 → 探索低效、学习不稳定。
  - 提出 **EFGCL**：**guided RL**，在训练中引入**外部辅助力**；灵感来自艺术体操 **spotting**，使智能体**不依赖任务专用 reward shaping 或参考轨迹**即可反复体验成功执行。
  - 实验：**四足** Jump / Backflip / Lateral-Flip；EFGCL 将 Jump 学习加速约 **2 倍**，并使常规 RL **难以学会**的复杂全身动作可习得；仿真策略**直接迁移**到真机 KLEIYN，动作与仿真一致。
  - 机制解释：早期成功体验 → **Critic 价值估计更快对齐**「好状态」分布（页面给出与基线的迭代数对比叙述）。

## 「How EFGCL Works」摘录（页面小节）

### 1. Assistive Force（辅助力）

- 训练早期施加**较强的外部辅助力**，类比教练托举体操运动员，使机器人**从一开始就能完成动作**。
- 通过**频繁经历成功轨迹**，智能体快速学到**哪些状态有价值**，从而加速 **Critic / 价值函数**学习。

### 2. Curriculum Decay（课程衰减）

- 辅助**非永久**：随表现提升，按**成功率**自动**减小辅助力幅度**，避免策略**过度依赖**外力。
- 训练末期**完全撤除**辅助，策略**独立**执行动态动作。

## 其它页面块（标题级）

- **Simulation Results**：Jump；Backflip；Lateral-Flip；强调**仅用简单稀疏奖励**即可，无需复杂任务专用 shaping 或参考轨迹。
- **From Simulation to the Real World**：仿真策略直接部署到 KLEIYN，复现三种动态动作。
- **Accelerating Value Estimation**：将成功体验与 **Critic 收敛速度**关联（页面给出与基线的迭代数对比口径）。

## BibTeX（页面提供）

```bibtex
@article{yoneda2026efgcl,
    author={Keita Yoneda and Kento Kawaharazuka and Kei Okada},
    journal={IEEE Robotics and Automation Letters},
    title={EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force-Guided Curriculum Learning},
    year={2026},
    volume={11},
    number={5},
    pages={5907-5913},
    keywords={Learning from Experience; Bioinspired Robot Learning; Legged Robots},
    doi={10.1109/LRA.2026.3675955}
}
```

## 对 wiki 的映射

- 升格页面：[wiki/methods/efgcl.md](../../wiki/methods/efgcl.md)

## 参考链接（索引）

- 项目主页：<https://keitayoneda.github.io/kleiyn-efgcl/>
- DOI：<https://doi.org/10.1109/LRA.2026.3675955>
