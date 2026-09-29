# OWASP Mobile Application Security（mas.owasp.org）

> 来源归档

- **标题：** OWASP Mobile Application Security (MAS) — 旗舰项目门户
- **类型：** site（标准 + 测试指南门户）
- **来源：** OWASP Foundation
- **链接：**
  - 门户：https://mas.owasp.org/
  - MASVS：https://mas.owasp.org/MASVS/
  - MASTG：https://mas.owasp.org/MASTG/
  - MASWE：https://mas.owasp.org/MASWE/
- **代码：**
  - MASVS：https://github.com/OWASP/owasp-masvs（归档：[repos/owasp-masvs.md](../repos/owasp-masvs.md)）
  - MASTG：https://github.com/OWASP/owasp-mastg（归档：[repos/owasp-mastg.md](../repos/owasp-mastg.md)）
  - MASWE：https://github.com/OWASP/maswe（归档：[repos/owasp-maswe.md](../repos/owasp-maswe.md)）
- **入库日期：** 2026-09-29
- **一句话说明：** 移动应用安全的行业事实标准组合：**MASVS**（验证标准）+ **MASWE**（弱点枚举）+ **MASTG**（测试指南），供架构、开发与渗透测试对齐「测什么、怎么测、如何合规」。
- **沉淀到 wiki：** 是 → [`wiki/entities/owasp-mas.md`](../../wiki/entities/owasp-mas.md)

## 为什么值得保留

- 具身 / 机器人产品常附带 **Android/iOS  companion App**（遥操作、地图、固件 OTA、账号与设备绑定）；云边 API 安全见 [software-security-basics](../../wiki/concepts/software-security-basics.md)，**客户端面**需单独基线。
- MAS 与 **OWASP ASVS**（通用 Web/应用验证标准）互补：ASVS 不覆盖移动端特有攻击面（本地存储、越狱/root、平台 API、逆向韧性等）。
- MASVS 被多家平台与政府机构引用为移动安全与合规对照清单（门户「Trusted By」区列示 adopters）。

## 开源核查（2026-09-29）

| 项 | 状态 |
|----|------|
| 文档站 | **公开可读**（mas.owasp.org；内容源对应 GitHub 仓） |
| MASVS / MASTG / MASWE | **已开源** — 见对应 `sources/repos/` 归档 |

## 核心摘录

### 使命（门户）

> Define the industry standard for mobile application security.

三件套分工：

| 组件 | 全称 | 角色 |
|------|------|------|
| **MASVS** | Mobile Application Security Verification Standard | **应满足的控制项**（架构师 / 开发 / 测试共同语言） |
| **MASWE** | Mobile Application Security Weakness Enumeration | **常见弱点目录**（连接标准与测试用例） |
| **MASTG** | Mobile Application Security Testing Guide | **流程、工具、技术细节与测试用例**（可重复、可审计的验证方法） |

### MASVS 控制组（2026 站点结构）

标准按攻击面划分为带 `MASVS-*` 前缀的控制组：

| 组 | 主题 |
|----|------|
| MASVS-STORAGE | 设备上敏感数据的安全存储（静态数据） |
| MASVS-CRYPTO | 保护敏感数据的密码学用法 |
| MASVS-AUTH | 移动应用的认证与授权机制 |
| MASVS-NETWORK | 应用与远端之间的安全通信（传输中数据） |
| MASVS-PLATFORM | 与移动平台及其他已安装应用的安全交互 |
| MASVS-CODE | 数据处理安全实践与更新机制 |
| MASVS-RESILIENCE | 对逆向工程与篡改的韧性 |
| MASVS-PRIVACY | 保护用户隐私的控制项 |

### 与机器人场景的映射（归纳）

| 机器人产品形态 | 常触发的 MASVS 组 |
|----------------|-------------------|
| 遥操作 / 状态监控 App | NETWORK、AUTH、PLATFORM |
| 本地缓存地图、日志、凭证 | STORAGE、CRYPTO |
| 绑定机器人 SN / 家庭 Wi-Fi 凭证 | AUTH、STORAGE |
| 可 sideload 的 OEM 控制 App | CODE、RESILIENCE |
| 采集用户位置 / 视频预览 | PRIVACY |

## 对 wiki 的映射

- 实体：[owasp-mas](../../wiki/entities/owasp-mas.md)
- 概念：[software-security-basics](../../wiki/concepts/software-security-basics.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [systems_engineering_deploy_obs_security_primary_refs.md](./systems_engineering_deploy_obs_security_primary_refs.md) | 已列 OWASP **ASVS**；本页补 **移动客户端** 标准 |
| [edge-cloud-robotics](../../wiki/concepts/edge-cloud-robotics.md) | 边云通道安全；MAS 覆盖 **手机侧** 实现 |
| [model-versioning-ota](../../wiki/concepts/model-versioning-ota.md) | 机载 OTA 验签；移动 App 分发渠道另需 CODE / RESILIENCE 对照 |

## 推荐继续阅读

- MAS 门户：<https://mas.owasp.org/>
- MASVS 下载与索引：<https://mas.owasp.org/MASVS/>
- MASTG 目录：<https://mas.owasp.org/MASTG/>
