# video-prediction-policy

> 来源归档（国内具身开源全景）

- **标题：** video-prediction-policy
- **类型：** repo
- **机构：** 星动纪元
- **链接：** https://github.com/roboterax/video-prediction-policy
- **分类：** VLA/操作模型
- **入库日期：** 2026-09-06
- **一句话说明：** 星动纪元 开源项目 video-prediction-policy（VLA/操作模型），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/paper-shenlan-wm-02-vpp.md`](../../wiki/entities/paper-shenlan-wm-02-vpp.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/paper-shenlan-wm-02-vpp.md](../../wiki/entities/paper-shenlan-wm-02-vpp.md)

## 官方 README 补核（2026-10-08）

- **README：** https://github.com/roboterax/video-prediction-policy/blob/main/README.md
- **项目页：** https://video-prediction-policy.github.io/
- **论文：** https://arxiv.org/abs/2412.14803 （v1：2024-12-19）
- **机构：** 清华大学、加州大学伯克利分校、上海人工智能实验室、上海期智研究院、星动纪元；联合研究，不只署名公司。

两阶段入口为 `step1_prepare_latent_data.py` / `step1_train_svd.py` 和 `step2_train_action_calvin.py` / `step2_train_action_xbot.py`；评测用 `policy_evaluation/calvin_evaluate.py`，真机适配用 `step3_deploy_real_xbot.py`。

README 链接 `yjguo/svd-robot`、`yjguo/svd-robot-calvin-ft`、`yjguo/dp-calvin` 权重和 `yjguo/vpp_svd_latent` 潜数据；CLIP 基础编码器与 CALVIN 原始数据来自外部项目。开放代码和部分模型/潜数据，不代表全量自采 XHAND 原始数据开放。README 报告 CALVIN ABC 平均链长 4.33，训练配置为单节点 8 张 A800/H100；本轮核查公开入口，未重跑训练或下载大型资产。

项目页归档：[video-prediction-policy](../sites/video-prediction-policy.md)。
