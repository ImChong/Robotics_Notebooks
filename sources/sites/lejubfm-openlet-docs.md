# LejuBFM 官方文档：OpenLET 数据集与运控部署

> 来源归档（官方文档）

- **数据集说明：** <https://www.lejubfm.lejurobot.com/docs/datasets/openlet>
- **BFM 环境配置：** <https://www.lejubfm.lejurobot.com/docs/bfm/setup>
- **离线轨迹部署：** <https://www.lejubfm.lejurobot.com/docs/bfm/deployment>
- **代码仓库：** <https://atomgit.com/OpenLET/Lejulab_ARM>
- **入库日期：** 2026-10-11
- **一句话说明：** 乐聚官方小脑动作数据与通用运控基模说明：OpenLET 动作数据申请、鲁班机型重定向与采样，以及 sonic/heft 模型的编译和部署环境。

## OpenLET 文档要点

- 数据由乐聚与青瞳视觉联合打造，分非交互动作、环境交互、物体交互、武术和舞蹈五类。
- 动作重定向至鲁班机型，并按 120 fps 重采样，用于运控训练与评测。
- 数据需通过 OpenLET 平台申请；申请条款和审批结果以平台为准。文档提供可访问后使用的 `LET-HumanMotion-Roban2` 技术仓。
- 页面未展示动作条数统计。

## BFM 环境配置要点

- `sonic` 与 `heft` 模型共用一套依赖/编译环境，之后需在部署文档中选相匹配的模型、配置与轨迹。
- 官方步骤面向 **Roban 2.2**（`ROBOT_VERSION=17`），真机为 RK3588 / ARM64、gcc/g++ 11；仿真使用 Docker 与 MuJoCo。
- 仓库切换版本后需重新编译。仅更换同名 YAML、ONNX、CSV 可不重编，但需重启控制器。
- 该说明有明确机型边界，不能把参数直接套用到 Kuavo 或其他人形平台。

## 对 wiki 的映射

- [OpenLET LET-HumanMotion-Roban2](../../wiki/entities/openlet-humanmotion-roban2.md)
- [Lejulab_ARM 仓库归档](../repos/lejulab-arm.md)
- [OpenLET 社区](../../wiki/entities/openlet.md)
- [LeTools](../../wiki/entities/letools.md)
