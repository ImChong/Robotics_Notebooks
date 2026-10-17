# CoSMO 项目页（yangming911.github.io/cosmo-page）

> 来源归档

- **标题：** CoSMO: Conformal Semantic Mapping for Open-World Safe Robot Navigation
- **类型：** site（项目页）
- **URL：** <https://yangming911.github.io/cosmo-page/>
- **论文：** <https://arxiv.org/abs/2610.18321> — 归档见 [`sources/papers/cosmo_arxiv_2610_18321.md`](../papers/cosmo_arxiv_2610_18321.md)（项目页本身仍标 "Coming soon"，未链 arXiv）
- **代码：** <https://github.com/Yangming911/CoSMO> — 归档见 [`sources/repos/cosmo.md`](../repos/cosmo.md)
- **视频：** <https://www.youtube.com/watch?v=LvttKb8bCho>
- **入库日期：** 2026-10-17
- **一句话说明：** 展示 MP3D 仿真对比表、延迟消融与 Go2 真机覆盖率。

## 页面要点（2026-10-17 核查）

| 方法（MP3D，2,195 episode） | SR | SCR | SPL |
|------|----|----|----|
| OneMap 纯几何 | 11.3% | 25.5% | 0.071 |
| GCLIP Argmax | 18.4% | 13.1% | 0.103 |
| Uniform Dilation | 8.7% | 8.9% | 0.053 |
| CoSMO 开放词表 | 13.8% | 9.8% | 0.081 |
| CoSMO 已知词表 | 10.4% | 9.4% | 0.059 |
| CoSMO Grounding DINO 专家 | 16.5% | 10.5% | 0.097 |

- 延迟消融：330 episode，延迟 0–100 步，SCR 10.0%–10.9%。
- Go2 真机 30 次：覆盖率 90.73%（目标 90%），SCR 16.67%，SR 46.67%，专家反馈延迟 128.1–447.6 ms。
