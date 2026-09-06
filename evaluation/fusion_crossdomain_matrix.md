# CFAD 中文域零样本复验 · 信道退化矩阵汇总

> 生成：2026-09-06 09:45 · 脚本 `scripts/eval_fusion_crossdomain.py --deg <ch>`
> 方法：融合器在英文 ASVspoof train 上拟合（不碰中文集），CFAD 各布局全量 2000 条零样本应用

## 结果表（最优 = XLS-R 冻结 + LR，各方案 EER%）

| 布局 | 说明 | AASIST 单模型 | XLS-R+LR | 等权平均 | LR stacking |
|---|---|---|---|---|---|
| 原始 | clean | 44.20 | **15.50** | 19.00 | 18.30 |
| deg_amr | AMR 编码退化 | 45.80 | **15.60** | 40.40 | 38.90 |
| deg_mp3_16k | MP3 16k 压缩 | 45.00 | **17.40** | 19.60 | 19.30 |
| deg_noise | 加噪 | 46.50 | **19.30** | 35.40 | 33.90 |
| deg_phone8k | 电话 8k 带宽 | 45.50 | **16.20** | 29.10 | 27.60 |

（退化布局 n=2000；主布局 n=2000 = 1000 真 + 1000 伪）

## 关键观察

1. **AASIST 单模型在 CFAD 全布局约 44-47%（≈随机）**——英文域训练模型对中文真伪完全失去判别力
   （对照：英文 dev 真 0.000/伪 0.994 判别正常；中文 CFAD 真 0.116/伪 0.172 均判 bonafide）。
2. **XLS-R 冻结 + LR 全布局最优（15.5~19.3%）**——自监督多语表征是唯一对中文域保留判别力的前端。
3. 退化信道损失排序：**noise（19.3%）> mp3_16k（17.4%）> phone8k（16.2%）≈ amr（15.6%）≈ 原始（15.5%）**——
   加噪最伤语义前端，压缩/带宽类损失较小。
4. ⚠️ 全部布局均未达计划书 §一「中文域 EER<5%」验收线——量化证明：**纯英文训练的融合器无法满足中文域要求，
   需中文域适配（增量微调/中文伪数据训练）**，这正是中文域作为参赛卖点的数据支撑。

## 部署影响（对接 reverse-screen 与红队质量）

- reverse-screen 全量跑出 hard_examples=2546/clean 3117（82%）——根因是 AASIST 对中文红队合成音频**全量漏检**
  （edgetts 100%、sovits 92% hard，score 0.01~0.2），**并非难例价值高，而是 AASIST 中文域失效**。
- 结论：红队音频的「难度打标」不能用英文 AASIST 单通道，应改用 **XLS-R+LR（中文域最优）** 或融合栈重打分；
  quality_gate 的 reverse-screen 需接入语义通道替代 acoustic-only。

## 文件

- 单布局报告：`evaluation/fusion_crossdomain.json` / `_deg_{amr,mp3_16k,noise,phone8k}.json`（+ .md 同对）
- 特征缓存：`.tmp_ssl/crossdomain/xlsr_cfad*_2000.npz`、`aasist_*_2000.npz`（复用秒级）
