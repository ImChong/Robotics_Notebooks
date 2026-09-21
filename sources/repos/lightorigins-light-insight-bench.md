# Light-INSIGHT-Bench（lightorigins/Light-INSIGHT-Bench）

- **URL：** <https://github.com/lightorigins/Light-INSIGHT-Bench>
- **组织：** lightorigins
- **许可证：** Apache-2.0
- **关联项目页：** [INSIGHT-Bench 项目页](../sites/light-insight-bench.md)
- **关联论文：** [LightNav-0 arXiv:2608.30935](../papers/lightnav0_arxiv_2608_30935.md)

## 一句话说明

LightNav-0 官方 object-goal 导航评测：Isaac Sim 5.1 + Isaac Lab 2.3.2 runner、7 策略 adapter、`insight-bench verify` evidence pack 与 PR 驱动 leaderboard。

## 运行时入口（README 口径）

| 步骤 | 入口 |
|------|------|
| 安装 SDK | `$ISAACLAB_DIR/isaaclab.sh -p -m pip install -e .` |
| 拉取 episodes | `hf download LightOriginsHQ/light-insight-bench --repo-type dataset --local-dir $DATA_DIR` |
| 数据校验 | `isaaclab.sh -p -m insight_bench check-data --episodes ... --scene-root ...` |
| 评测 LightNav-0 | `bash scripts/eval_lightnav0.sh` |
| 自定义策略 | `bash scripts/eval_custom.sh` + `policies/` 模板 |
| 证据打包 | `insight_bench pack` → `insight_bench verify evidence-pack.zip` |

## 交叉链接

- [INSIGHT-Bench 实体](../../wiki/entities/insight-bench.md)
- [LightNav-0 论文实体](../../wiki/entities/paper-lightnav-0.md)
