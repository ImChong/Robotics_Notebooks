# PushMimic 官方项目与交互仿真 / Official Project and Interactive Simulator

- **原址：** https://silvery107.github.io/pushmimic/
- **核对日期：** 2026-11-26
- **核对范围：** 作者项目首页、真机结果说明、视频链接及在线 demo 的 `demo.js` 和 `sim.js` 静态阅读，未运行网页仿真。
- **通读：** 已完成

## 公开演示与复现的区别

- [交互入口](https://silvery107.github.io/pushmimic/) 可查看质量和坡度改变后策略的姿态；[作者视频](https://youtu.be/ap9OXna8eHo)提供方法与实机演示。
- [demo.js](https://silvery107.github.io/pushmimic/static/js/demo/demo.js) 读取部署到站点的 MuJoCo 模型、元数据、已导出的 policy 和名义参考；[sim.js](https://silvery107.github.io/pushmimic/static/js/demo/sim.js) 实现观察、前向、PD 子步，页面用 Three.js 渲染。坡度通过重力旋转模拟，渲染再将场景转回坡面视图；滑块改变仿真，不把质量或坡度寄存量直接送 actor。
- 浏览器程序是仿真单段参考回放，不是论文中重复步态的真机跑台运行。公开可访问不等于有明确开源许可，也不能据此声称训练、SBTO 数据及真机部署栈已经发布。

- **Wiki：** [PushMimic](../../wiki/entities/paper-pushmimic.md)
