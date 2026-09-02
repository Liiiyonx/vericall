# 谛听 VeriCall · 跨域评测矩阵（P2-2）

- 生成时间：2026-09-02 17:08:01
- 训练域：2019LA（现模型，AASIST，dev EER 0.745% / eval EER 3.49%）

> 本矩阵回答评委对「跨语言/跨信道泛化」的核心疑问。每个测试格 = 英文训练模型在该域的 EER；泛化鸿沟 = 该格 EER − 英文 eval EER。
> 缺权重或数据时不编造数字，标 TODO 并附填充流程。

## 矩阵（行=训练域，列=测试域）

| 训练域 \ 测试域 | 2019LA eval（英） | FMFCC-A（中） | CFAD（中） | 红队干净 | 红队电话 | In-the-Wild |
|---|---|---|---|---|---|---|
| 2019LA（现模型） | 3.49% | TODO | TODO | TODO | TODO | TODO |

## 各域状态与填充流程

| 测试域 | 数据可用 | 说明 | 结果 |
|---|---|---|---|
| 2019LA eval（英） | ✅ | eval 协议 | 已知 3.49% |
| FMFCC-A（中） | ❌ | 缺 dev 协议/flac（先跑 convert_fmfcc_protocol.py） | TODO |
| CFAD（中） | ❌ | 缺 dev 协议/flac（先跑 convert_fmfcc_protocol.py） | TODO |
| 红队干净 | ❌ | 缺红队音频（P2-1） | TODO |
| 红队电话 | ❌ | 缺红队音频（P2-1） | TODO |
| In-the-Wild | ❌ | 缺 In-the-Wild 数据 | TODO |

## 即插即用填充步骤（真实数据到位后）

1. **放权重**：把训练好的 `best.pth` 放进 `external/aasist/exp_result/<实验>/weights/`（含 `config.conf`）。
2. **放英文域**：`VERICALL_ASVSPOOF_LA` 指向 ASVspoof2019 LA（已有）。
3. **放中文域**：
   ```
   python scripts/convert_fmfcc_protocol.py --root <FMFCC-A> \
           --out data/raw/FMFCC-A_asvspoof
   # CFAD 同理 -> data/raw/CFAD_asvspoof
   ```
4. **放红队音频**：真实方言克隆音频（P2-1 GPT-SoVITS 合成）放入 `data/redteam/`。
5. **跑本脚本**：`python evaluation/cross_domain_eval.py` → 自动计算可算的格并刷新本矩阵。

## 泛化鸿沟（待填充）

泛化鸿沟 = 各中文/红队域 EER − 英文 eval 3.49%。**英文训练→多域测试的劣化规律**本身就是挑战杯/大创的研究叙事（方案书阶段二：把劣化变成研究点）。
