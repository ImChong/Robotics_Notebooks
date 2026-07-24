# GB/T 43200-2023 机器人一体化关节性能及试验方法

> 来源归档（ingest）

- **标题：** GB/T 43200-2023《机器人一体化关节性能及试验方法》
- **类型：** standard（国家标准）
- **英文名称：** Performance and related test methods of mechatronic joints for robots
- **标准状态：** 现行
- **发布 / 实施：** 2023-09-07 / 2024-04-01
- **发布单位：** 国家市场监督管理总局、国家标准化管理委员会
- **主管部门 / 归口：** 中国机械工业联合会；SAC/TC 591（全国机器人标准化技术委员会）
- **分类号：** CCS J28；ICS 25.040.30
- **官方入口：**
  - [开放国家标准信息 · GB/T 43200-2023](https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=B2E40B3445ACE9E166E8E402E89853AF)
  - [全国标准信息公共服务平台详情](https://std.samr.gov.cn/gb/search/gbDetailed?id=053404E3EE3F8F91E06397BE0A0A9209)
- **入库日期：** 2026-07-24
- **一句话说明：** 规定协作机器人与腿足式机器人 **一体化关节** 的性能指标与试验方法，是国内关节模组测功/对拖验收的主对标国标。
- **沉淀到 wiki：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

## 为什么值得保留

- 人形 / 协作关节验收要从「电机峰值扭矩宣传」落到 **模组级可复现试验项**；本标准给出性能与方法框架。
- 国内测功机与关节测试系统厂商（如 AIP）明确宣称方案对标本标准；缺少该条一手标准链接会导致 wiki 只能引用二手产品页。

## 范围与适用（公开摘要）

| 项 | 内容 |
|----|------|
| 对象 | 机器人 **一体化（机电）关节** |
| 适用 | 协作机器人、腿足式机器人关节；其他类型参照执行 |
| 内容性质 | 性能规定 + **试验方法** 描述 |

> 全文为受版权保护的国家标准文本；本仓库只保留元数据、公开平台链接与对 wiki 的映射，不转载标准正文条款。

## 与测功机台架的关系

- 试验需在可控负载下测量扭矩、转速、定位/控制响应与温升等——工程实现上对应 **测功机 / 对拖台架 + 扭矩传感器 + 功率分析 / 采集**。
- 验收粒度是 **关节模组**，不是裸电机；与 [电机设计流程](../../wiki/overview/motor-design-workflow.md) 步骤 8「整机 TN + 总线力矩模式」对齐。

## 对 wiki 的映射

- [电机测功机（Dynamometer）](../../wiki/concepts/motor-dynamometer.md)
- [电机设计流程](../../wiki/overview/motor-design-workflow.md)
- [力矩电机设计纵深 Stage 6](../../roadmap/depth-torque-motor-design.md)

## 相关一手索引

- [motor_dynamometer_primary_refs.md](motor_dynamometer_primary_refs.md)
