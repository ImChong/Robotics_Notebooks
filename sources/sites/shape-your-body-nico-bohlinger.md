# Shape Your Body — 项目页归档

> 来源归档（ingest 辅助）

- **类型：** site / project page
- **URL：** <https://nico-bohlinger.github.io/shape-your-body/>
- **关联论文：** [shape_your_body_arxiv_2606_00702.md](../papers/shape_your_body_arxiv_2606_00702.md)
- **PDF 镜像：** <https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/shape_your_body.pdf>
- **入库日期：** 2026-06-02
- **一句话说明：** 论文配套站点：交互式 URMA 策略演示、VGDS 共设计对比滑条、方法动画与引用块。

## 页面要点（与 PDF 交叉核对）

- **TL;DR：** 训练一个多具身策略与价值函数，再用价值梯度在数分钟内设计数百台新机器人。
- **Try It Yourself：** 可选机器人（Unitree Go2、MIT Humanoid、Golem、ANYmal C、Booster T1、Mini PI、Fourier GR1-T2 等）；Reference vs Co-Design；种子 0–9；VGDS 迭代 0/50；速度指令 Forward/Sideways/Turning；本地浏览器跑 URMA 策略。
- **VGDS 公式：** $\hat{J}_\lambda(f)$ 为状态库上 critic 均值减 $\lambda$ 倍归一化 $\|f-f_{\mathrm{ref}}\|_2^2$；Adam 步 + trust-region clip + $[-1,1]$ clip。
- **训练集：** 50 机器人（15 四足 / 31 双足与人形 / 4 六足）；每机 190–1177 连续设计参数；质量、惯量、几何、关节限位、PD、执行器属性等随机化。
- **效率叙事：** 单机器人 RL 共设计每个初值需新训练；VGDS 训练一次（约 7–9 h）后每设计约 1–2 min。
- **Citation：**
  ```bibtex
  @article{bohlinger2026shape,
      title={Shape Your Body: Value Gradients for Multi-Embodiment Robot Design},
      author={Bohlinger, Nico and Peters, Jan},
      year={2026}
  }
  ```
- **致谢 / 资助：** NCN Poland OPUS Weave UMO-2021/43/I/ST6/02711；DFG PE 2315/17-1。
- **致谢站点灵感：** Kevin Zakka、Brent Yi、Younghyo Park；基于 *One Policy to Run Them All* 站点风格。

## 对 wiki 的映射

- [paper-shape-your-body-value-gradient-design.md](../../wiki/entities/paper-shape-your-body-value-gradient-design.md) — 交互演示与公开表述
