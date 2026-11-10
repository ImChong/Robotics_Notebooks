## [2026-11-10] ingest | ResGAC（arXiv:2610.09479）全文复核与原节点升级

- **对象：** Precise SE(3) End-Effector Tracking in Whole-Body Humanoid Control / 全身人形控制中的精确 SE(3) 末端执行器跟踪。
- **去重：** 保留唯一实体 `wiki/entities/paper-resgac-precise-se3-end-effector-tracking.md`；论文、官网继续指向同一实体，不新建重复项目节点。
- **一手来源：** https://arxiv.org/html/2610.09479v1 ，已核读正文摘要、I–V（方法公式 1–30、实验 Tables I–IV 和结论）；https://github.com/ResGAC/ResGAC-website 中 README.md/index.html/details.html 已核查站点展示和补充参数。
- **修订：** 补通读标记与原文 original 字段；中英双标题、阅读骨架、GAC 无实测 wrench 的边界、控制公式、六阶段训练和真实实验条件、重画 Mermaid 算法流程图；修正被误标为原文直引的改写摘要。
- **开源状态：** 官网注明 Code 将在论文被接收后开放；当前只有项目展示网页源文件，无可核验的控制器训练/部署入口，源码 sequenceDiagram 暂不适用。
- **工程与评测边界：** 记录 G1 50 Hz、FastSAC、4096 环境、L40S、OptiTrack 外部定位/配准、插孔 18/20 与 H0 对不同轴不同程度改善。
- **派生文件：** 未修改 `catalog.md`、`log.md` 或前端导出文件，等待主分支自动生成。
