# SpaceX 官方页面：Falcon 9 回收硬件与 Starship 第五次试飞助推器塔臂捕获

> 来源归档（ingest）

- **Falcon 9 车辆页：** <https://www.spacex.com/vehicles/falcon-9/>
- **Starship 第五次试飞页：** <https://www.spacex.com/launches/starship-flight-5>
- **类型：** 官方网站
- **入库日期：** 2026-11-06
- **核对范围：** 两页均为前端渲染，用 headless Chromium 渲染后读取全文（2026-11-06）。页面只描述硬件与任务过程，不披露制导、导航与控制算法。
- **通读：** 已完成

## 核心摘录

1. **Falcon 9 定位。** "Falcon 9 is the world's first orbital class reusable rocket. Reusability allows SpaceX to refly the most expensive parts of the rocket"。一级为九台 Merlin 发动机，页面另列"LANDING LEGS"栏目。
2. **栅格舵。** "Falcon 9 is equipped with four hypersonic grid fins positioned at the base of the interstage. They orient the rocket during reentry by moving the center of pressure."
3. **Merlin。** "The Merlin engine was originally designed for recovery and reuse."
4. **Starship 第五次试飞（2024-10-13）。** 助推器在上升、级间分离、返场点火（boostback burn）和滑行后执行着陆点火，"was caught by the chopstick arms of the launch and catch tower at Starbase"；页面称捕获前须满足 "Thousands of distinct vehicle and pad criteria"，且首次尝试即成功。

## 与已有资料的关系

- 制导层面的公开描述见 [Blackmore 2016](../papers/blackmore_autonomous_precision_landing_space_rockets.md)（F9R 返回剖面与着陆点火、CVXGEN）。
- 本页只作为硬件与里程碑的一手出处，不据此推断控制算法。

## 对 wiki 的映射

- [动力下降制导（Powered Descent Guidance）](../../wiki/concepts/powered-descent-guidance.md) — 助推器全剖面回收
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 6
