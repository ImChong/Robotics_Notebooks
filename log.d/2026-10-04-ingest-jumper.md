## [2026-10-04] ingest | KingKongRobotics/jumper — 六足机器人训练至部署链路

- 核对公开仓库、官方产品页、硬件文档、项目指南、训练教程、仿真设计和部署说明；训练/仿真/导出代码已公开，第三方许可证由 NOTICE 和各上游许可约束。
- 新增仓库与产品页来源档案，提炼 Jumper 实体页，展开 tripod 速度跟踪、jump 参考残差、dance 轨迹评分、双 MuJoCo 后端、ONNX/layout 校验和 bundle/FSM 路径。
- 对照 Inspect Robots 详情页与 markdown 渲染器，发现页面使用 ```mermaid 围栏而渲染器只识别标准三反引号围栏；同步修复两张图的围栏，并让新页使用可渲染格式。
- 明确 Jumper 的 RKNN 板端推理和 1 kHz 电机总线闭环尚未验证，不把脚本打包能力当作真机验证。
