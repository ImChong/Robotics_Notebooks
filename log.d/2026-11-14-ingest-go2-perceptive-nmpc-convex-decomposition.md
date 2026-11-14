## [2026-11-14] ingest | sources/blogs/wechat_guyueju_perceptive_nmpc_convex_decomposition_2026-11-14.md — 感知型 NMPC 高程图凸分解与 Go2 源码核对

- **意图：** 消化用户提供的古月居技术解读 PDF，建立可溯源的感知型 NMPC / 凸踏面几何约束工程入口。
- **开源核验：** Grandia 等原始方法见 IEEE T-RO 2023（ANYmal）；Go2 适配补丁和 ROS 2/MuJoCo 工具见 Khaledsweid1/ocs2-perceptive-mpc-go2（公开）。
- **重要边界：** Go2 工程的落足区域约束默认关闭，指南说明尚不能完整爬完楼梯；模拟射线高程图非真机感知，附件参数来自文章与仓库文档，未独立复测。
- **沉淀：** [Go2 感知型 NMPC 实体](../wiki/entities/ocs2-perceptive-mpc-go2.md)，并接入 [感知越障纵深路线](../roadmap/depth-perceptive-locomotion.md)。
