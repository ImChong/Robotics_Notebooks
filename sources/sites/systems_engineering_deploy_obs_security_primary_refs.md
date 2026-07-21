# 容器部署、可观测性与软件安全一手资料索引

> 来源归档（ingest）

- **标题：** Docker / Kubernetes / CI·CD、Logs·Metrics·Tracing、认证授权密钥与供应链安全一手依据
- **类型：** official docs / standard（合集）
- **入库日期：** 2026-07-21
- **一句话说明：** 支撑机器人训练集群、仿真 farm、边缘网关与发布流水线的部署与安全基线。
- **沉淀到 wiki：** 是 → [container-orchestration-cicd](../../wiki/concepts/container-orchestration-cicd.md)、[observability-logs-metrics-tracing](../../wiki/concepts/observability-logs-metrics-tracing.md)、[software-security-basics](../../wiki/concepts/software-security-basics.md)

## 核心摘录

### 1) Docker 与 OCI 容器

- **来源：** [Docker Docs](https://docs.docker.com/)；[OCI Runtime Spec](https://github.com/opencontainers/runtime-spec)
- **要点：** 镜像分层、只读 rootfs、cgroup 资源限制；GPU 容器依赖 NVIDIA Container Toolkit。训练环境可复现的第一层往往是「镜像 digest 锁定」。
- **对 wiki 的映射：** [container-orchestration-cicd](../../wiki/concepts/container-orchestration-cicd.md)

### 2) Kubernetes

- **来源：** [Kubernetes Documentation](https://kubernetes.io/docs/home/)（Pod、Deployment、Service、Job、DaemonSet）
- **要点：** 声明式期望状态；适合仿真 farm / 批训练 / 遥测聚合；**不**适合把 1 kHz 力矩环塞进普通 Pod 共享内核调度。
- **对 wiki 的映射：** [container-orchestration-cicd](../../wiki/concepts/container-orchestration-cicd.md)、[edge-cloud-robotics](../../wiki/concepts/edge-cloud-robotics.md)

### 3) CI/CD

- **来源：** GitHub Actions / GitLab CI 文档；CNCF 对软件交付流水线的惯例
- **要点：** lint → 测试 → 构建镜像 → 签名 → 部署；机器人额外需要 **硬件在环 / Sim2Sim 门禁** 与模型制品晋升。
- **对 wiki 的映射：** [container-orchestration-cicd](../../wiki/concepts/container-orchestration-cicd.md)、[model-versioning-ota](../../wiki/concepts/model-versioning-ota.md)

### 4) 可观测性：Logs / Metrics / Tracing

- **来源：** [OpenTelemetry](https://opentelemetry.io/docs/)；Google SRE Book（Monitoring Distributed Systems）
- **要点：**
  - **Logs**：事件叙述；运控侧应异步、限速，避免同步写盘。
  - **Metrics**：计数/直方图/仪表；控制频率、deadline miss、总线利用率是机器人关键指标。
  - **Tracing**：跨服务请求因果链；云边 API 有用，硬实时环路通常用周期时间戳直方图替代全链路 span。
- **对 wiki 的映射：** [observability-logs-metrics-tracing](../../wiki/concepts/observability-logs-metrics-tracing.md)

### 5) 安全：认证、授权、密钥、供应链

- **来源：**
  - [OWASP ASVS](https://owasp.org/www-project-application-security-verification-standard/)
  - [NIST SP 800-63](https://pages.nist.gov/800-63-3/)（数字身份）
  - [SLSA](https://slsa.dev/) / [Sigstore](https://www.sigstore.dev/)（供应链与签名）
  - OAuth 2.0 / OIDC（[RFC 6749](https://www.rfc-editor.org/rfc/rfc6749)、[OpenID Connect Core](https://openid.net/specs/openid-connect-core-1_0.html)）
- **要点：**
  - **认证 (AuthN)** 回答「你是谁」；**授权 (AuthZ)** 回答「你能做什么」。
  - 密钥进 **KMS/HSM/密封密钥**，禁止硬编码进镜像；OTA 与模型制品须签名验签。
  - 供应链：锁定依赖、SBOM、镜像签名、最小权限 CI。
- **对 wiki 的映射：** [software-security-basics](../../wiki/concepts/software-security-basics.md)、[model-versioning-ota](../../wiki/concepts/model-versioning-ota.md)

## 当前提炼状态

- [x] 摘要与 wiki 映射
