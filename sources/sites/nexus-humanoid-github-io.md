# NEXUS — 项目页归档

- **来源：** https://nexus-humanoid.github.io/
- **类型：** site
- **机构：** 上海交通大学（SJTU）；作者主页指向 Xiangyu Miao（缪翔宇），导师 Weinan Zhang（张伟楠）
- **状态：** Research Preview（截至 2026-09-06）
- **归档日期：** 2026-09-06

## 一句话说明

NEXUS（**N**eural e**X**ecution for **U**nified whole-body teleoperation **S**ystem 的缩写待论文确认）定位为 **感知型基础策略（perceptive foundation policy）**：把 **实时人体运动** 映射为 **跨域、地形感知的人形全身遥操作行为**；口号「Live human motion in. Terrain-aware humanoid behavior out.」

## 项目页核查（步骤 2.5）

| 资源 | 项目页状态 | 核查结论 |
|------|------------|----------|
| Paper | **Coming Soon**（按钮 `aria-disabled`） | **待发布** |
| arXiv | **Coming Soon** | **待发布** |
| Code | **Coming Soon** | **训练/部署代码待发布** |
| Video / Preview | **Coming Soon** | 仅有 `preview-poster.webp` 静态海报 |
| GitHub | 页内无可点击 Code 链 | 仅发现组织 [nexus-humanoid/nexus-humanoid.github.io](https://github.com/nexus-humanoid/nexus-humanoid.github.io)（**静态站源码**，非策略训练栈） |
| X (Twitter) | [xiangyu_miao 推文](https://x.com/xiangyu_miao/status/2095706449028710582) | 宣传入口 |

**开源结论：** 截至入库日 **未开源** 可复现训练/推理代码；勿将 GitHub Pages 仓误认为算法实现。

## 从项目页可确认的技术线索

- **任务：** Cross-Domain **Whole-Body Teleoperation**（跨域全身遥操作）
- **方法定位：** **Perceptive Foundation Policy** — 统一 **视觉感知** 与 **全身控制**
- **行为：** **Terrain-aware**（地形感知）；预览海报 alt 文本提及 **simulated and real stair terrain**
- **输入/输出叙事：** 人体 live motion → 人形 terrain-adaptive 行为

## 作者侧公开信息（个人主页交叉）

- [xiangyumiao.pages.dev](https://xiangyumiao.pages.dev/)：SJTU 硕士（Weinan Zhang 组）；研究方向含 **foundation policies that unify perception and whole-body control**；NEXUS 列为 **Selected work · Coming Soon**

## 交叉链接

- Wiki：[wiki/entities/nexus-humanoid.md](../../wiki/entities/nexus-humanoid.md)
- 仓库归档（仅静态站）：[sources/repos/nexus-humanoid-github-io.md](../repos/nexus-humanoid-github-io.md)
- 相近路线：[Perceptive BFM](../../wiki/entities/paper-perceptive-bfm.md)（地形感知 + raw 运动参考）
