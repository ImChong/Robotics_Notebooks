---
type: concept
tags: [systems-engineering, security, authn, authz, secrets, supply-chain]
status: complete
updated: 2026-07-21
related:
  - ./model-versioning-ota.md
  - ./container-orchestration-cicd.md
  - ./edge-cloud-robotics.md
  - ../overview/topic-systems-engineering.md
sources:
  - ../../sources/sites/systems_engineering_deploy_obs_security_primary_refs.md
summary: "软件安全基础（身份认证、授权、密钥管理、供应链安全）：机器人云边通道与制品发布的最小安全基线。"
---

# 软件安全基础（认证 / 授权 / 密钥 / 供应链）

## 一句话定义

**软件安全基础** 区分「你是谁 / 你能做什么 / 密钥如何保管 / 构建链是否可信」——覆盖遥操作、OTA、模型仓库与 CI 制品。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AuthN | Authentication | 身份认证 |
| AuthZ | Authorization | 授权 |
| KMS | Key Management Service | 密钥管理服务 |
| OIDC | OpenID Connect | 基于 OAuth2 的身份层 |
| SBOM | Software Bill of Materials | 软件物料清单 |
| SLSA | Supply-chain Levels for Software Artifacts | 供应链安全框架 |

## 为什么重要

- 未签名模型 OTA = 任意人可改机器人行为。
- CI 密钥泄漏会污染整个训练与发布链。
- 遥操作通道缺 TLS/鉴权等于把控制权暴露在网上。

## 核心原理

1. **AuthN**：密码、密钥、OIDC/mTLS 证明身份。
2. **AuthZ**：RBAC/ABAC 限制动作（谁能对哪台机器人下发部署）。
3. **密钥管理**：密钥进 KMS/HSM 或密封；短期凭证；轮换。
4. **供应链**：锁定依赖、生成 SBOM、镜像/模型签名（Sigstore 等）、最小权限 CI。

## 工程实践

- 机载只存验证公钥；私钥留在签名服务。
- 分环境凭证（dev/stage/prod）；禁止把 cloud key 打进机器人镜像。
- PR 流水线跑漏洞扫描与许可证检查；发布需人工或策略门禁。
- 与 [OTA](./model-versioning-ota.md) 联动：验签失败 → 拒绝更新并告警。

## 局限与风险

- 「内网」不是安全边界——机器人常漫游到陌生 Wi-Fi。
- 过度复杂的 PKI 无人维护 = 过期证书导致全队停机。

## 关联页面

- [模型版本管理与 OTA](./model-versioning-ota.md)
- [容器编排与 CI/CD](./container-orchestration-cicd.md)
- [边缘–云端协同](./edge-cloud-robotics.md)

## 参考来源

- [部署可观测安全一手资料](../../sources/sites/systems_engineering_deploy_obs_security_primary_refs.md)

## 推荐继续阅读

- OWASP ASVS；SLSA：<https://slsa.dev/>
