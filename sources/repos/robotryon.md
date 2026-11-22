# RoboTryOn 官方源码与复现入口

> 来源归档（repo）；对应唯一论文详情 [paper-robotryon](../../wiki/entities/paper-robotryon.md)。

- **标题：** RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On
- **类型：** repo
- **官方仓库：** <https://github.com/ryan-seungyong-lee/RoboTryOn>
- **论文：** <https://arxiv.org/abs/2611.05163>
- **项目页：** <https://ryan-seungyong-lee.github.io/RoboTryOn/>
- **模型权重：** <https://huggingface.co/RyanL22/RoboTryOn>
- **入库日期：** 2026-11-05
- **复核日期：** 2026-11-22
- **核对范围：** 官方 README（Overview、Installation、Inference、End-to-end、Stage by stage、Training、Adding an embodiment）；`scripts/run_pipeline.py` 与 `configs/pipeline.yaml`。
- **通读：** 已完成

## 官方代码运行入口

| 阶段 | 已核实入口 | 输入 / 输出 |
|---|---|---|
| 人视频预处理 | `preprocessing/prepare_human_clip.py` | 人视频、HaWoR、SAM3、目标机器人配置 → episode、手部骨架、遮罩、关键帧 |
| 图像关键帧转换 | `scripts/infer_keyframes.py` | episode、参考图、图像权重 → robot keyframe PNG |
| 视频合成 | `scripts/infer_video.py` | 关键帧、骨架流、video LoRA → robot video MP4 |
| 动作标注 | `scripts/label_actions.py` | 合成视频、IDM 权重 → actions.npy / states.npy |
| 腕部校正 | `scripts/correct_wrists.py` | 手腕重建 + 本体配置 + 动作 → 校正后的动作和状态 |
| 端到端编排 | `scripts/run_pipeline.py --config configs/pipeline.yaml` | 按 stages: preprocess / keyframes / video / actions 组织各子进程 |
| 训练 | `scripts/train_image.py`、`scripts/train_video.py`、`scripts/train_idm.py` | 现有机器人遥操作数据；具体配置在 `configs/` |

## 官方运行步骤（改写自 README，须自行提供本体配置/视频）

```bash
git clone https://github.com/ryan-seungyong-lee/RoboTryOn
cd RoboTryOn
# 三类模型建议使用独立 Python 环境，见 README
hf download RyanL22/RoboTryOn --local-dir ckpt
python scripts/run_pipeline.py --config configs/pipeline.yaml \
  embodiment=path/to/my_embodiment.yaml idm_embodiment=my_embodiment \
  input.videos.left=human_left.mp4 input.videos.right=human_right.mp4 \
  input.caption="put the ball in the box" input.object_prompts='[ball]' \
  reference.left=ref_left.png reference.right=ref_right.png \
  checkpoints.image=ckpt/image checkpoints.video=ckpt/video checkpoints.idm=ckpt/idm \
  hawor_root=third_party/HaWoR output_dir=runs/demo
```

**输出：** `runs/demo/episode`、`keyframes`、`video`、`actions/actions.npy`、`actions/states.npy`。代码和权重已公布；HaWoR / SAM3 需另行安装，机器人 URDF/MJCF 本体资产、原始人视频与实际控制环境不随此仓库分发。这里核查了源码入口，**没有宣称已在本机执行推理或复现实机成功率**。

## 与论文映射

- [原始论文归档](../papers/robotryon_arxiv_2611_05163.md)
- [官方项目主页归档](../sites/robotryon.md)
- [wiki 唯一论文详情](../../wiki/entities/paper-robotryon.md)
