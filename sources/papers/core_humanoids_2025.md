# CoRe（Humanoids 2025）

> 来源归档（ingest）

- **标题：** CoRe: A Hybrid Approach of Contact-Aware Optimization and Learning for Humanoid Robot Motions
- **类型：** paper
- **来源：** 项目页摘要 / IEEE Xplore 元数据 / 官方软件 README
- **原始链接：**
  - 项目页 <https://tmjeong1103.github.io/CoRe-page/>
  - IEEE <https://doi.org/10.1109/Humanoids65713.2025.11203055>
  - 代码 <https://github.com/tmjeong1103/CoRe>
- **作者：** Taemoon Jeong†、Yoonbyung Chai†、Sol Choi、Jaewan Bak、Chanwoo Kim、Jihwan Yoon、Yisoo Lee、Jongwon Lee、Kyungjae Lee、Joohyung Kim、Sungjoon Choi\*
- **机构：** 高丽大学（Korea University）；韩国科学技术研究院（KIST）；伊利诺伊大学厄巴纳-香槟分校（UIUC）
- **会议：** 2025 IEEE-RAS 24th International Conference on Humanoid Robots（Humanoids），Seoul，pp. 293–300
- **入库日期：** 2026-08-15
- **一句话说明：** 在 RL 跟踪之前，用接触段检测 + 接触约束轨迹优化 + 足朝向与自碰平滑，把文本生成的人体运动修成可执行参考，再以接触感知奖励训策略；跨全身 / 轮式 / 上身人形、无需逐任务调参。

## 核心摘录

### 1) 问题

文生运动已能出像人的轨迹，但直接喂 RL 会把 **初始运动学不可行**（脚滑、浮空、关节加速度过大）留给策略去「补课」，表现为不稳。现有路线多半 **只靠 RL、不先修参考**。

### 2) 管线

1. 自然语言 → 人体运动生成
2. 机型相关重定向（软件侧对应 RMR / DMR）
3. **Contact-aware motion Refinement（CoRe）**
4. 接触感知奖励的物理模仿 RL → sim-to-real

精炼四步：接触段检测（趾轨迹）→ 接触约束轨迹优化 → 足偏航调整 → 自碰处理与平滑。

### 3) 评测（项目页）

- 三类具身：全身人形、轮式人形、上身人形
- 声称 **无任务特定调参、无动力学级优化** 即可迁移
- 场景从简单上身手势到复杂全身 locomotion；展示 sim-to-real

### 4) 开源边界

- **已开源：** 重定向 + 接触精炼产品 [tmjeong1103/CoRe v0.1.0](https://github.com/tmjeong1103/CoRe/releases/tag/v0.1.0)（Apache-2.0）
- **未随仓发布：** 论文中的 text-to-motion 生成器与 contact-aware RL 训练代码
- **无 arXiv：** 截至入库日以项目页 + IEEE 为准

```bibtex
@inproceedings{jeong2025core,
  author    = {Jeong, Taemoon and Chai, Yoonbyung and Choi, Sol and
               Bak, Jaewan and Kim, Chanwoo and Yoon, Jihwan and
               Lee, Yisoo and Lee, Jongwon and Lee, Kyungjae and
               Kim, Joohyung and Choi, Sungjoon},
  title     = {CoRe: A Hybrid Approach of Contact-Aware Optimization
               and Learning for Humanoid Robot Motions},
  booktitle = {2025 IEEE-RAS 24th International Conference on
               Humanoid Robots (Humanoids)},
  year      = {2025},
  pages     = {293--300},
  doi       = {10.1109/Humanoids65713.2025.11203055}
}
```

## 对 wiki 的映射

- 升格 [CoRe 论文实体](../../wiki/entities/paper-core.md)
- 软件 [CoRe v0.1.0](../../wiki/entities/core-retarget.md)
- 前端重定向对照 [RMR](../../wiki/entities/paper-rmr.md)
- 交叉 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)

## 当前提炼状态

- [x] 项目页摘要 + 四步精炼 + 开源边界
- [x] 与 RMR / GMR / Kimodo / SOMA 栈互链
- [ ] IEEE 全文表格数字（付费墙；有预印本后再补）
