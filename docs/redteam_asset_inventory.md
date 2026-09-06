# 红队伪造音频资产盘点（v0.8-redteam 语料快照）

> 生成：2026-09-06 · 数据源 `data/redteam/factory/meta.csv`（4509 行）
> 用途：语料评审、材料引用、增量训练数据侧的权威盘点。

## 一、总量与构成

| 维度 | 明细 | 条数 |
|---|---|---|
| **信道** | clean（母本） | 3117 |
| | 退化 phone8k / mp3_16k / amr / noise | 各 348（1392） |
| **引擎** | Edge-TTS（8 音色 × 6 方言谱系） | 4069 |
| | GPT-SoVITS v2 克隆（自举 11 参考音） | 440 |
| **方言** | mandarin 1466 / sichuan 763 / cantonese 760 / minnan 758 / dongbei 381 / henan 381 | 4509 |
| **质量门禁** | 通过 | 3982 |
| | 隔离（超长 >20s，可切分救回） | 527 |

对应磁盘 wav：factory/ 下约 **4165**（clean 3117 = 2679 edgetts + 440 sovits + 重复/退化版本计 4509 meta 行）。

## 二、clean 母本难度分布（XLS-R + 中文域 LR，AUC 0.950）

| 档位 | 判定 | 条数 | 占比 |
|---|---|---|---|
| <0.3 | **隐蔽（难检）** | 3090 | 99.1% |
| 0.3~0.5 | 边界 | 26 | 0.8% |
| ≥0.5 | 可检 | 1 | 0.03% |

- 难度分均值 0.078（edgetts 0.077 / sovits 0.081）——**红队合成对中文域最强检测器几乎全漏检**；
- 详表：`data/redteam/factory/redteam_difficulty.csv`（含 engine/dialect/script_id/分数）。

## 三、质量门禁（quarantine 527 条）

- 全部为 `duration_ok=False`（>20s 上限，含 14~73s 母本与 22~73s 克隆长句）；
- 处置：按详册 §7.2 3~6s 切分后可救回（edgetts 语速快、切 6s 段语义完整度高；sovits 语速慢需 8~10s 段）；
- 清单：`data/redteam/factory/quarantine.csv`。

## 四、法律与合规（三条铁律核对）

1. ✅ 话术全部为**脱敏合成改写**（无人名/证件号/卡号真实信息），语料原始文本不直接对外；
2. ✅ 音频为 AI 合成（Edge-TTS 服务条款：仅研究评测用途；GPT-SoVITS 本地合成），无真人声纹；
3. ⚠️ 红队集 **禁入训练管线**（详册 §7.3）——仅作评测/难例分析，增量训练需另取合规中文伪样本（见下）。

## 五、增量训练数据侧的现实状态（重要）

| 数据源 | 真/伪 | 本地状态 | 可用性 |
|---|---|---|---|
| FMFCC-A | 1 万真(N01 系) + 4 万伪(A01-A13 系) | **17,636 伪 wav**（A 系，全 label 0） | **合规中文伪样本池（非自产，可入训练）**；真 N01 待下载 |
| CFAD | 1000 真 + 1000 伪 | 完整（2000 条 + 4 退化布局） | **评测集**，按隔离纪律不作训练 |
| 红队 | 全伪 | 4605 wav | **禁入训练**，作评测难例集 |
| ASVspoof LA | 英文真伪 | 完整（train/dev/eval） | 英文基线已用 |

**结论（2026-09-06 修正）**：中文域增量训练数据侧——
- **伪侧已足**：FMFCC-A 17.6k 伪样本（A 系攻击）可入训练管线（合规、非自产），红队仅作评测；
- **缺真侧**：需补 FMFCC-A N01 真样本（1 万）或采用其它中文真语料（aishell1 等；注意 CFAD 真样本是评测集不能挪作训练）；
- 真伪齐备后即可把中文域模型（XLS-R+中文LR / 融合栈）从 CFAD EER 15.5% 方向训练拉向 <5%。

## 六、抽检/评审状态

- LLM 预筛：677 条 → **30 条可疑候选**（`evaluation/review_prefilter_suspect.csv`，drop 率 3.4% 远低于 30% 重做线）；
- 人工评审工作表（含全文 + LLM 建议）：`evaluation/review_30candidates_worksheet.csv`（30 行待填 human_verdict）；
- 抽检原表：`data/scam_corpus/review_sample.csv`（433 行，verdict 列待填）；
- benign 侧 529 条（6 类含近边界）评审表：`data/scam_corpus/benign_review_sample.csv`。
