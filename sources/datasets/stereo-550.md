# Stereo-550 / Ego-OSCAR-550h（Hugging Face）

> 来源归档（ingest 配套数据集）

- **标题：** Stereo-550（论文称 Ego-OSCAR-550h）
- **类型：** dataset
- **链接：** <https://huggingface.co/datasets/fpvlabs/stereo-550>
- **论文：** [arXiv:2608.08285](https://arxiv.org/abs/2608.08285)
- **项目页 / 3D：** <https://www.fpvlabs.ai/ego-oscar/cap>
- **机构：** 第一人称视觉实验室（FPV Labs）
- **许可：** FPV Labs 定制 license（`fpvlabs-license`；见 <https://fpvlabs.ai/license> / HF 卡内链接）；**gated** 申请制
- **入库日期：** 2026-08-12
- **一句话说明：** ~550 h/相机标定第一人称立体 RGB + 稠密自由格式动作字幕 +（多数会话）同步 6 轴 IMU；为众包可复现 ego 采集硬件的规模验证语料。

## 规模速查（HF card / 论文）

| 项 | 数值 |
|----|------|
| 时长 | **≈550 h / 相机**（≈1,100 stereo cam-h） |
| 会话 | **1,462** stereo sessions（左右各一，共 2,924 视频文件） |
| 动作段 | **209,315**；时间线覆盖 ~100%；中位 94 段/会话 |
| 词汇 | 460 verbs；32,630 object phrases；57,104 verb–object 组合 |
| 长尾 | Top-20 表达式仅占 1.5% 实例 |
| IMU | 1,271/1,462 会话（86.9%）；论文 ~120 Hz；HF 目录说明写 ~180 Hz |
| 分辨率 | 1280×720 @ 30 fps，H.264 MP4 |
| 标定 | 每会话 `calibration.json`（针孔 + radtan；基线 ≈42 mm） |
| 贡献者 / 设备 | 25 user IDs / 13 共享设备 |
| 访问 | **gated** + 定制研究许可 |

## 每会话目录（HF card）

```
<user_id>/<session_id>/
  *_left.mp4 / *_right.mp4
  *_action_labels.json
  calibration.json
  imu/.../imu_synced.csv   # 可选
```

> **注：** 论文附录还描述全库 WiLoR 手重建 JSON；截至入库日 HF README 目录树**未列出**手重建文件——下载后以实际 release 为准。

## 对 wiki 的映射

- [Ego-OSCAR 论文实体](../../wiki/entities/paper-ego-oscar.md)
- [论文归档](../papers/ego_oscar_arxiv_2608_08285.md)
- [项目页归档](../sites/fpvlabs-ego-oscar.md)
- [Ego4D](../../wiki/entities/paper-ego4d.md)、[Ego 数据采集分类](../../wiki/overview/ego-category-01-data-collection.md)
