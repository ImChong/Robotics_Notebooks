# ultra-humanoid.github.io（ULTRA 项目页）

> 来源归档（ingest）

- **标题：** ULTRA — UIUC
- **类型：** site / project-page
- **官方入口：** <https://ultra-humanoid.github.io/>
- **入库日期：** 2026-09-29
- **一句话说明：** IROS 2026 **移动操作最佳论文入围** 配套站：同一套权重支持 **稠密 MoCap 参考跟踪**、**稀疏键盘/点击目标** 与 **第一人称深度** 下的长时域全身 loco-manipulation；页内提供 **MuJoCo 浏览器交互 demo**、真机户外/室内视频与 BibTeX。

## 页面公开资源（检索自 2026-09-29）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://ultra-humanoid.github.io/> |
| 论文 abs | <https://arxiv.org/abs/2603.03279> |
| PDF | <https://arxiv.org/pdf/2603.03279> |
| **代码** | <https://github.com/Sirui-Xu/ULTRA>（页头 GitHub 按钮；**已开源**） |

## 开源核查结论

- **代码：** **已开源** — [Sirui-Xu/ULTRA](https://github.com/Sirui-Xu/ULTRA)（Apache-2.0；InterMimic 衍生栈 MIT；含 Isaac Gym 训练 / MuJoCo sim2sim / 真机部署脚本与预置 teacher 权重）。
- **数据：** Google Drive 发布 **InterMimic 格式 OMOMO 参考** 与 **ULTRA 重定向+增广 G1 轨迹**（见仓库 README）；非 Hugging Face 托管。
- **权重：** 仓库含 `ultra/weights/teacher_ultra_inference.pth`；student 需按 README 训练或 finetune。

## 对 wiki 的映射

- [`wiki/entities/paper-notebook-ultra-unified-multimodal-control-for-autonomous.md`](../../wiki/entities/paper-notebook-ultra-unified-multimodal-control-for-autonomous.md)
- [`wiki/tasks/loco-manipulation.md`](../../wiki/tasks/loco-manipulation.md)
- [`sources/repos/ultra-humanoid.md`](../repos/ultra-humanoid.md)

## BibTeX（站点页提供）

```bibtex
@inproceedings{he2026ultra,
  title     = {ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation},
  author    = {He, Xialin and Xu, Sirui and Li, Xinyao and Dong, Runpei and Bian, Liuyu and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
