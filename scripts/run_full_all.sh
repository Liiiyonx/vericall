#!/usr/bin/env bash
# 全量 clean（62,999）**十列**总编排：AASIST 5 -> RawNet2 3 -> XLS-R 2，串行不抢 GPU。
#
# 覆盖论文 Table 1/4 全部 10 列（M0/M1/M1'/LP/LP-FT/RN2-EN/RN2-CN/RN2-LP/M2/M3）。
# 导师要求十列全量，使正文可写"full-partition re-scoring reproduces every quoted value"，
# 且全表口径一致（不出现"部分列全量、部分列子样"的自曝）。
#
# 顺序刻意把**最贵的 M2/M3（4.3h/列）放在最后**：任何时刻中断都已拿到前八列，
# 前八列即可支撑 Table 1 的核心结论与 Table 3/4 的全部数字。
#
# 为什么必须串行：2026-09-14 首轮失败的原因就是 5 个模型并行抢 8G 卡 + 超时被杀。
#
# 用法：bash scripts/run_full_all.sh          # 后台跑
#      tail -f reports/full_all.log          # 看进度
#
# 幂等：已存在的输出（>1MB）自动跳过，可反复运行续跑；失败列不阻断后续。

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
PY="D:/VeriCall_data/conda_envs/vericall/python.exe"
LOG="$ROOT/reports/full_all.log"
REP="$ROOT/evaluation/reports"

mkdir -p "$ROOT/reports"
exec > >(tee -a "$LOG") 2>&1

echo "================================================"
echo "全量 clean 十列编排启动 $(date '+%F %T')"
echo "根目录: $ROOT"
echo "预计总耗时 ~14.3h（AASIST 3.6 + RawNet2 3 + XLS-R 8.6）"
echo "================================================"

# ---------- 阶段 1：AASIST 五列 ----------
echo
echo ">>> 阶段 1/3：AASIST 五列（M0/M1/M1p/LP/LP-FT）"
"$PY" scripts/run_full_clean_scores.py
echo "<<< 阶段 1 结束 $(date '+%T')"

# ---------- 阶段 2：RawNet2 三列 ----------
echo
echo ">>> 阶段 2/3：RawNet2 三列（RN2-EN/RN2-CN/RN2-LP）"
"$PY" scripts/run_full_remaining_scores.py
echo "<<< 阶段 2 结束 $(date '+%T')"

# ---------- 阶段 3：XLS-R 两列（最贵，放最后） ----------
echo
echo ">>> 阶段 3/3：XLS-R 两列（M2/M3，各约 4.3h）"
for sc in v4 wide; do
  OUT="$REP/cfad_scores_xlsr_${sc}_clean_full.jsonl"
  if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    echo "[skip] xlsr_$sc: 已完成 $(basename "$OUT")"
    continue
  fi
  echo "----- XLS-R scorer=$sc -----"
  "$PY" evaluation/eval_xlsr_full.py --scorer "$sc" || echo "[FAIL] xlsr_$sc（继续下一列）"
done
echo "<<< 阶段 3 结束 $(date '+%T')"

# ---------- 汇总 ----------
echo
echo "================================================"
echo "完成情况 $(date '+%F %T')"
echo "================================================"
for t in en0 cn_ft m1lp lp aasist_lpft rn2_en0 rn2_cn rn2_lp xlsr_v4 xlsr_wide; do
  f="$REP/cfad_scores_clean_${t}_full.jsonl"
  [ -f "$f" ] && printf "  %-14s %6d / 62999\n" "$t" "$(wc -l < "$f")" \
               || printf "  %-14s %s\n" "$t" "未生成"
done

echo
echo ">>> 一致性核验（子样 vs 全量）"
"$PY" evaluation/verify_full_vs_subset.py
echo
echo "全部结束 $(date '+%F %T')"
