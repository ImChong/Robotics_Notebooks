# Humanoid-UniTracker 项目页（yinkangning0124.github.io）

> 来源归档（ingest）

- **标题：** UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots
- **类型：** site / project-page
- **官方入口：** <https://yinkangning0124.github.io/Humanoid-UniTracker/>
- **arXiv：** <https://arxiv.org/abs/2507.07356>
- **GitHub（页面源码）：** <https://github.com/yinkangning0124/Humanoid-UniTracker>
- **入库日期：** 2026-09-18
- **一句话说明：** 论文配套站点：强调三阶段框架（teacher → CVAE universal student → adaptation）、G1 真机多样动作与 text-to-motion 演示；**Code 链注释未启用**，训练代码未发布。

## 页面公开信息（检索自 2026-09-18）

| 资源 | URL / 状态 |
|------|------------|
| 项目首页 | <https://yinkangning0124.github.io/Humanoid-UniTracker/> |
| arXiv | <https://arxiv.org/abs/2507.07356> |
| Code | **未启用**（HTML 中 Code 区块被注释；无独立训练仓库链接） |
| GitHub | <https://github.com/yinkangning0124/Humanoid-UniTracker>（Nerfies 模板 + 静态资源，非 RL 代码） |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **核心贡献：** 特权 teacher → CVAE universal student（partial/full 观测对齐 latent）→（项目页文案）轻量 adaptation 微调难序列。
2. **动机：** MLP 在部分观测下表达力不足、朝向漂移、OOD 泛化差。
3. **平台：** Unitree G1 真机；含拉伸、武术、舞蹈、高踢、踢球、深蹲等演示。
4. **Text-to-Motion：** 页面展示 MDM 文本 prompt → 机器人执行（squat / punch / waltz 等）。
5. **机构：** SJTU、Shanghai AI Lab、Shanghai Innovation Institute、PKU、ZJU、Fudan、HKUST-GZ、ShanghaiTech。

## 开源核查结论

- **代码开放程度：确认未开源**（截至 2026-09-18）。
- 项目页 Footer 指向 Nerfies 网站模板；GitHub 仓无 `train.py` / IsaacGym 配置 / checkpoint。
- 论文 PDF 未给出可下载权重 URL。

## 对 wiki 的映射

- [`wiki/entities/paper-loco-manip-161-024-unitracker.md`](../../wiki/entities/paper-loco-manip-161-024-unitracker.md)
- [`sources/papers/unitracker_arxiv_2507_07356.md`](../papers/unitracker_arxiv_2507_07356.md)
- [`sources/repos/humanoid-unitracker.md`](../repos/humanoid-unitracker.md)
