# brave-eai/DreamTrue

- **类型：** 官方仓库
- **仓库：** [brave-eai/DreamTrue](https://github.com/brave-eai/DreamTrue)
- **论文：** [arXiv:2610.12468](https://arxiv.org/abs/2610.12468)
- **项目页：** [DreamTrue](https://brave-eai.github.io/DreamTrue/)
- **数据/权重：** [ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue)
- **核查日期：** 2026-10-11
- **开放状态：** 部分开源。README 列出 calibration、wmvideo、reward 代码及部分数据；完整数据将在后续发布。

## 模块

- `calibration`：动作/相机几何校准。
- `wmvideo`：视频世界模型。
- `reward`：具身视频奖励模型。

ModelScope 有数据/权重入口不等于完整训练资产全部齐备。

- [论文摘录](../papers/dreamtrue_arxiv_2610_12468.md)
- [项目页归档](../sites/dreamtrue-project-page.md)
- [Wiki 实体](../../wiki/entities/paper-dreamtrue-action-faithful-world-model.md)

## 2026-11-13 静态源码复核

- **固定提交：** `991e2963c015be9c5b7f21695bb82bfaaceffed8`。
- **核对范围：** 根与calibration/reward/wmvideo README全文；三视角shell入口和 `inference_demo_v2v_vace.py` 的初始化、双LoRA、generate、run_inference条件组装/保存路径；未运行、未阅读全库训练代码。
- **真实入口：** 在wmvideo目录用 `Inference/V2V_VACE/inference_threeviews_mixed_20260623.sh`，传本地base、双LoRA、条件mini包和上游数据路径。入口可dry-run组装命令；正式通过Accelerate运行demo脚本。
- 脚本从manifest/dataset config读取样本，每视角仅首帧作为reference，渲染动作RGB/depth及camera参数作为未来条件；三视角顺序统一。`WanV2VVaceInferenceModule` 先enable_plucker再载LoRA/geometry全量键，`generate`调用WanVideoPipeline，按padding信息裁帧并保存视频和JSON记录。
- `--use_plucker`必须匹配checkpoint的mask_cam_embedding，缺键或错误开关会警告并可能使用随机geometry权重。推理summary的success_rate是文件生成成功率，非机器人任务成功率。
- **开放范围：** 三组件有公共接口，ModelScope有双LoRA/reward/condition样例链接；wmvideo无训练代码，calibration README仍列full pipeline/SAM3/full data待发布，reward全量标注待发布。本次仅确认入口，未下载或验证资产可运行性。
