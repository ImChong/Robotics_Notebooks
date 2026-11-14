# EmbodiedCity/WorldVLN（代码仓库）

> 来源归档

- **名称：** WorldVLN
- **类型：** repo
- **组织：** EmbodiedCity
- **官方链接（项目页）：** <https://github.com/EmbodiedCity/WorldVLN.code>
- **GitHub 组织页同名仓：** <https://github.com/EmbodiedCity/WorldVLN>
- **论文：** <https://arxiv.org/abs/2605.15964>
- **项目页：** <https://embodiedcity.github.io/WorldVLN/>
- **入库日期：** 2026-05-24
- **一句话说明：** WorldVLN 官方代码入口； ingest 时仓库为新建占位状态，复现细节以论文 PDF 与项目页训练/评测说明为准。

## 维护备注

- 用户提供的链接后缀为 `.code`（与项目页 hero 区一致），非典型 `.git` 后缀；收录时保留官方 URL，并在论文 source 中注明组织下标准 GitHub 路径供检索。

## 对 wiki 的映射

- [WorldVLN（论文实体）](../../wiki/entities/paper-worldvln-aerial-vln-wam.md)
- [worldvln_arxiv_2605_15964.md](../papers/worldvln_arxiv_2605_15964.md)

## 2026-11-13 源码复核

- **固定提交：** `c45434f862a4742fbd0d4d560ed45cfd68049579`；当前已有可运行入口，替代早期“占位”判断。
- **核对范围：** README 全文、`infer/run_server.sh`/`config.json`、`infer/server.py` 的模型初始化、请求/会话管理、latent 推理、16-action 解码与返回路径；`Worldmodel/runtime/tools/infinity_streaming_session.py` 全文。未阅读全部训练代码，未下载权重或执行 GPU/模拟器。
- **入口：** `INFINITY_CKPT` 与 `STAGE2_LATENT2ACTION_CKPT` 指定权重，`bash infer/run_server.sh` 启动 Uvicorn `server:app`，默认 `tsformer_latent`。README 给 HF 模型入口、IndoorUAV/UAV-Flow 数据环境入口及两阶段训练命令；这是已发布接口核验，不是成功复现。
- **请求：** `/v1/predict_delta_actions` 接收 `session_id`、首帧与指令、后续实测帧；严格在线用 `allow_future_segments=1`，否则首帧仅 warmup 并等待更多观测。单帧+指令默认重置会话。
- **世界路径：** `_infer_summed_codes_for_step` 将已收到的真实 prefix 编码，经 `gen_one_example(return_summed_code_only=True)` 预测；`gt_leak` 是已观测 token 的条件注入，不等于输入未来真值。清除预测 cache，真实 prefix 单独管理。
- **动作路径：** `_stage2_predict_16_actions_for_segment_cm_deg` 将 summed codes patchify、VAE feature adapter→TimesFormer 四帧滑窗→重叠平均/反归一化，返回16条。API顺序 `[dx,dy,dz,droll,dyaw,dpitch]`，cm/deg；内部角度/平移顺序不同。
- 默认49帧配置只覆盖三个段；持续长程任务需明确会话推进策略。调试缓存目录默认启用时会解码/保存视频，不能笼统称服务完全不做像素解码。
- README 声明 CC BY4.0，并列出上游依赖；其许可不自动覆盖外部数据/模型。当前部分训练入口和精度配置晚于论文，应分别记录版本。
