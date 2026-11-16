# HumanNet 原始 Ego 视频：ModelScope 入口核验记录

> 来源归档（dataset link audit）。记录用户提供的数据入口与两篇官方论文的关系；**尚未读取到 ModelScope 数据卡或文件清单，不把时长和下载权限视为已核实**。

- **HumanNet 原始 Ego 入口（待核验）：** <https://www.modelscope.cn/datasets/DAGroup-PKU/HumanNet/tree/master/data>
- **HumanNet 论文：** <https://arxiv.org/html/2605.06747>
- **Open-HumanWAM 论文：** <https://arxiv.org/html/2611.06817v2>
- **官方代码：** <https://github.com/DAGroup-PKU/HumanNet>
- **Open-HumanWAM-50K HF 数据集：** <https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K>
- **Open-HumanWAM 官方 README 列出的 ModelScope 50K 镜像：** <https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3>
- **核对日期：** 2026-11-11
- **状态：** 两个 ModelScope 页面均无法通过当前工具获取；没有验证“20 万小时原始 Ego 数据”已开放可下载。

## 论文支持的规模与边界

| 资产 | 可核实统计 | 计数口径 |
| --- | --- | --- |
| HumanNet | 约 1,000,000 h | **第一+第三人称**人中心视频整体 |
| 进入 Open-HumanWAM 的 HumanNet 子集 | **90,275.25 h → 40,743.58 h** | 策展前 → 策展后，论文 v2 表 1；不是 HumanNet 全部 Ego 的规模 |
| Open-HumanWAM-50K | **104,289.41 h → 50,000.00 h** | 五个来源合计，策展前 → 策展后 |
| 所称“20 万小时原始 Ego” | **待验证** | 暂无 ModelScope 数据说明、文件清单及许可证佐证 |

## 开放性与复现

- 官方 Open-HumanWAM 分支 README 另列出 50K 数据的 ModelScope 入口，不能证明其镜像内容完整，也不能代替 HumanNet 原始 Ego 数据集。
- Hugging Face 50K 数据卡为 gated，用户需要接受条件才能访问；源数据的许可证需分别核对。
- 上述链接属原始数据发现线索，不假定可以直接运行完整下载命令。

## 知识库映射

- [HumanNet 全库](../../wiki/entities/humannet.md)
- [Open-HumanWAM 论文与模型](../../wiki/entities/paper-open-humanwam.md)
- [Open-HumanWAM-50K 数据](open-humanwam-50k.md)
