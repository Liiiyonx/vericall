# A5 语义置信度校准 · 方案落档（2026-09-07）

> 目标：让 LLM 语义分数（risk 0-1）跨温度/样本可比，支撑融合器阈值与拦截后复盘。
> 状态：方案就绪；执行数据依赖下一次 F1/深评运行**逐条保存 risk**（eval_semantic_f1.py 下一版加 --save-probs 输出行级 id/risk/label）。

## 设计
1. **采集**：跑冻结评测集（2,868 条）保存每条 (id, 真值, risk, category)；温度 0.1（已固定低温，方差小）另抽 300 条温度 {0.0,0.4,0.8} 各一遍估方差；
2. **校准曲线**：risk 分桶（0.05 步）→ 桶内真实诈骗率 vs 均值 risk → 可靠性图（reliability diagram）；偏差大时温度缩放 T 使 ECE 最小；
3. **产出**：`evaluation/semantic_calibration.md`：可靠性图数据、最优 T、ECE；给融合器阈值建议（block ≥0.6 需校准后复核，与 0.55/0.35→0.52/0.44 的声纹做法同套路）；
4. **工具**：脚本复用 eval_semantic_f1 的调用器 + sklearn 可选（无则手算 ECE/可靠性）。

## 与声纹标定的对称性
声纹已有 DET/κ（voiceprint_calibration.md）；语义侧校准后即可回答评委"LLM 分数可信吗"，
并支撑 A3 复盘模板的 risk 口径（>0.5 判 block 的根据）。

## 待办
- [ ] eval_semantic_f1.py 增加 --save-probs（行级落盘）
- [ ] 跑一次全量取数（~5 分钟云端）
- [ ] 出可靠性图 + ECE + 温度缩放表
