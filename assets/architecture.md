# 谛听 VeriCall · 系统架构

```mermaid
flowchart LR
    A[来电音频] --> B{融合管线 pipeline}

    subgraph 通道① 声学伪造
        B --> C[AASIST<br/>合成/克隆概率]
    end
    subgraph 通道② 家庭声纹
        B --> D[CAMPPlus<br/>是否本人]
    end
    subgraph 通道③ 话术语义
        B --> E[SenseVoice ASR<br/>→ Ollama deepseek-r1]
        E -.Ollama 不可达.-> E2[规则评分器<br/>rule_scorer 兜底]
    end

    C --> F[融合编排器 FusionOrchestrator]
    D --> F
    E --> F
    E2 --> F

    F --> G{三级裁决}
    G -->|硬拦截 ≥0.85 & conf≥0.6| H[🔴 拦截]
    G -->|加权汇总 ≥0.70| H
    G -->|加权汇总 ≥0.50<br/>或单通道中危升级| I[🟡 警惕]
    G -->|其余| J[🟢 放行]

    H --> K[适老提示 / 子女卡片]
    I --> K
    J --> K
```

## 融合策略（可解释优先）

1. **纵深防御**：任一通道高置信危险（score≥0.85 且 confidence≥0.6）即直接拦截，单点失效不影响安全。
2. **置信度加权汇总**：三通道可疑度按权重（声学 0.35 / 声纹 0.30 / 话术 0.35）汇总，低置信通道贡献打折。
3. **单通道中危升级**：任一通道 score≥0.60 且 conf≥0.6，至少升为警惕，避免被其他通道稀释成放行。
4. **离线降级**：Ollama/SenseVoice 不可达时，话术通道走规则评分器，声学/声纹用缓存或 stub，演示链路不中断。

> 融合阈值 0.85 / 0.50 / 0.70 的标定实验见任务书 P1-5（当前为工程初值，后续以评测标定替换）。
