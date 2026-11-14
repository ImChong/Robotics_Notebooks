# HaoranZhuExplorer/AD-E2E-JEPA

> 来源归档（官方仓库一手资料）

- **类型：** official repository / research code
- **仓库：** <https://github.com/HaoranZhuExplorer/AD-E2E-JEPA>
- **README：** <https://github.com/HaoranZhuExplorer/AD-E2E-JEPA#readme>
- **论文：** <https://arxiv.org/abs/2609.34085>
- **核查日期：** 2026-10-11
- **对应实体：** [AD-E2E-JEPA](../../wiki/entities/paper-ad-e2e-jepa.md)
- **论文摘录：** [arXiv 归档](../papers/ad_e2e_jepa_arxiv_2609_34085.md)
- **一句话说明：** 作者指定的官方代码仓库；核查时只有 README 和 overview 图片，README 称可运行代码仍在准备中。

## 开源状态（2026-10-11）

**待发布 / 尚不可复现。** 仓库 README 自称官方源码，并链接论文；但 README 明确表示源码正在准备公开，预计在 2026-10-19 前发布。当前文件列表只有 README.md 与 imgs/，未见训练、评测、数据加载、推理或部署代码、安装说明、权重或数据。

这与 arXiv PDF 中 “source code is available” 的表述不一致。复现状态应按当前可访问仓库文件判断，不把论文中的承诺当成已经发布的可运行实现。

## 当前可核对内容

- README：项目名称、论文链接、方法总览图和待发布源码说明。
- imgs/overview.png：架构示意图，不是执行入口。
- 未发现可核验的训练脚本、规划脚本、依赖锁文件、数据预处理代码或模型权重。

## 复现边界

- 可依据论文方法和 NAVSIMv2 报告数字做分析，但暂时无法在官方仓库复核模型结构、数据管线、checkpoint、GPU 时延或表格结果。
- 论文报告的时延测量使用 NVIDIA A100 80 GB；源码公开后仍需验证硬件、候选并行策略和数据准备是否可复现。
- 后续核查应确认仓库 release/commit、license、依赖、权重与数据访问方式，以及是否提供训练和推理入口。

## 交叉来源

- 论文：[AD-E2E-JEPA arXiv 摘录](../papers/ad_e2e_jepa_arxiv_2609_34085.md)
- 结构化知识页：[AD-E2E-JEPA 实体页](../../wiki/entities/paper-ad-e2e-jepa.md)

## 2026-11-13 固定版本源码复核

- 版本：[dc122f69e1ad1bd5553d4ffbc74da20a047089ee](https://github.com/HaoranZhuExplorer/AD-E2E-JEPA/tree/dc122f69e1ad1bd5553d4ffbc74da20a047089ee)。已读根README、`docs/GCZSP_EVAL.md`、v3规划sbatch全文、`jepa_wm_projector.py`全文，`jepa_wm_model.py`构造/encode/predict/advance_rollout_context完整函数、agent构造与接口，v3 GCZSP脚本入口、场景/词表加载、完整BatchedWMUnroll/GCZSPEvalObjective及候选评分路径。
- README标注10/19代码、10/23权重已发布，11/19重构待办。当前含可运行实现，HF链接HaoranZhu/AD-E2E-JEPA；未下载权重或执行实验。
- sbatch→v3 GCZSP脚本→Hydra构造JEPA_WMAgent/JEPA_WMModel→载checkpoint→SceneLoader/metric cache→固定8192词表下采样→四帧图像编码与共享projector→8步predict/advance_rollout_context→终点MSE取argmin→PDM统计。
- 当前序列脚本同时选0.5/1/2/4s轨迹，EPDMS取4s，计时排除场景加载和后续评测、含预处理至选轨迹；默认五次warmup不计统计。sbatch依赖Singularity overlay、地图、cache绝对路径，不能直接复制为通用单机命令。
