#!/usr/bin/env bash
# 全量十列跑批状态快照（只读，供定时跟进用）。
# 用法：bash scripts/full_run_status.sh
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
REP="$ROOT/evaluation/reports"
PY="D:/VeriCall_data/conda_envs/vericall/python.exe"

echo "时间: $(date '+%F %T')"
echo "================================================================"
echo "十列全量 clean 打分状态"
echo "================================================================"

TARGET=62999
total_done=0
for t in en0 cn_ft m1lp lp aasist_lpft rn2_en0 rn2_cn rn2_lp; do
  f="$REP/cfad_scores_clean_${t}_full.jsonl"
  if [ -f "$f" ]; then
    n=$(wc -l < "$f" 2>/dev/null || echo 0)
    if [ "$n" -ge "$TARGET" ]; then st="DONE"; total_done=$((total_done+1)); else st="PARTIAL"; fi
  else
    n=0; st="--"
  fi
  printf "  %-14s %6d / %d  %s\n" "$t" "$n" "$TARGET" "$st"
done
for sc in v4 wide; do
  f="$REP/cfad_scores_xlsr_${sc}_clean_full.jsonl"
  if [ -f "$f" ]; then
    n=$(wc -l < "$f" 2>/dev/null || echo 0)
    if [ "$n" -ge "$TARGET" ]; then st="DONE"; total_done=$((total_done+1)); else st="PARTIAL"; fi
  else
    n=0; st="--"
  fi
  printf "  %-14s %6d / %d  %s\n" "xlsr_$sc" "$n" "$TARGET" "$st"
done
echo "----------------------------------------------------------------"
echo "已完成 $total_done / 10 列"

echo
echo "---- 进程/日志 ----"
for lg in reports/rn2_bg.log reports/xlsr_tail.log reports/full_all.log; do
  if [ -f "$ROOT/$lg" ]; then
    echo "[$lg]"
    tail -3 "$ROOT/$lg" | sed 's/^/    /'
  fi
done

echo
echo "---- GPU ----"
"$PY" -c "
import torch
if torch.cuda.is_available():
    free,total=torch.cuda.mem_get_info()
    print('  used %.2f GB / %.2f GB  (free %.2f GB)'%((total-free)/1e9,total/1e9,free/1e9))
else:
    print('  CUDA 不可用')
" 2>/dev/null || echo "  (torch 查询失败)"

echo
echo "---- 报告文件时间戳 ----"
ls -lt "$REP"/cfad_breakdown_*_full_*.json 2>/dev/null | head -5 | awk '{print "  "$6" "$7" "$8"  "$9}' || echo "  (无)"
