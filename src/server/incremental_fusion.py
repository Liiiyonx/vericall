# -*- coding: utf-8 -*-
"""
增量融合：每窗 EMA 平滑 + 迟滞状态机（绿→黄→红）。
================================================
设计要点（对应任务书 P1-1）：
  - 每通道分数 EMA 平滑：s_t = 0.6·score_t + 0.4·s_{t-1}（单窗误判不会瞬间翻转状态）；
  - 状态机带迟滞：
      绿→黄：需连续 2 窗 fused≥0.5（或单通道中危升级）；
      黄→红：fused≥0.7 或任一通道硬命中（score≥0.85 & conf≥0.6）；
      红→告警一次 + 10s 冷却（防反复轰炸老人）；连续 5 窗干净才降级回绿；
  - 复用 FusionOrchestrator.decide() 做每窗裁决，状态机只消费其输出，不重写融合逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fusion.fusion_orchestrator import FusionOrchestrator, ChannelVerdict

GREEN, YELLOW, RED = "GREEN", "YELLOW", "RED"
ADVICE = "建议先挂断，拨打 {家人} 平时的号码确认一下。"


@dataclass
class StreamState:
    state: str = GREEN
    fused: float = 0.0
    offline: bool = False
    last_transcript: str = ""


class IncrementalFusion:
    def __init__(self, orchestrator: Optional[FusionOrchestrator] = None,
                 ema_alpha: float = 0.6,
                 yellow_windows: int = 2,
                 recover_windows: int = 5,
                 cooldown_s: float = 10.0,
                 family_name: str = "家人"):
        self.orch = orchestrator or FusionOrchestrator()
        self.alpha = ema_alpha
        self.yellow_windows = yellow_windows
        self.recover_windows = recover_windows
        self.cooldown = cooldown_s
        self.family_name = family_name

        self.ema = {"acoustic": 0.0, "voiceprint": 0.0, "semantic": 0.0}
        self.state = GREEN
        self._alert_streak = 0   # 绿态下连续"黄/红需求"计数
        self._clean_streak = 0   # 连续"绿需求"计数
        self._last_alert_t = -1e9
        self.last_transcript = ""
        self.offline = False

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.ema = {"acoustic": 0.0, "voiceprint": 0.0, "semantic": 0.0}
        self.state = GREEN
        self._alert_streak = 0
        self._clean_streak = 0
        self._last_alert_t = -1e9
        self.last_transcript = ""

    # ------------------------------------------------------------------ #
    def update(self, acoustic: ChannelVerdict, voiceprint: ChannelVerdict,
               semantic: ChannelVerdict, t_rel: float, idx: int = 0
               ) -> list[dict]:
        # 1) EMA 平滑各通道分数
        for name, cv in (("acoustic", acoustic), ("voiceprint", voiceprint),
                         ("semantic", semantic)):
            self.ema[name] = self.alpha * cv.score + (1 - self.alpha) * self.ema[name]
            if cv.label not in ("stub", "unknown") or cv.detail:
                pass

        sac = ChannelVerdict("acoustic", round(self.ema["acoustic"], 3),
                             acoustic.label, acoustic.detail, acoustic.confidence)
        svp = ChannelVerdict("voiceprint", round(self.ema["voiceprint"], 3),
                             voiceprint.label, voiceprint.detail, voiceprint.confidence)
        ssem = ChannelVerdict("semantic", round(self.ema["semantic"], 3),
                              semantic.label, semantic.detail, semantic.confidence)

        fused = self.orch.decide(sac, svp, ssem)

        # 2) 计算"需求等级"
        #    注意：硬命中 / 单通道中危升级 用【原始】通道裁决（cv），不缓存 EMA。
        #    EMA 只平滑"融合分"以吸收单窗尖峰；而纵深防御（硬拦截）与中危升级
        #    必须基于通道模型当窗给出的原始风险，否则 0.9 的硬命中会被 EMA 拖到
        #    第 3 窗才越过 0.85，丧失"硬命中即拦"的语义。
        hard = (acoustic.score >= self.orch.hard_block and acoustic.confidence >= 0.6) or \
               (voiceprint.score >= self.orch.hard_block and voiceprint.confidence >= 0.6) or \
               (semantic.score >= self.orch.hard_block and semantic.confidence >= 0.6)
        single_esc = (acoustic.score >= 0.60 and acoustic.confidence >= 0.6) or \
                     (voiceprint.score >= 0.60 and voiceprint.confidence >= 0.6) or \
                     (semantic.score >= 0.60 and semantic.confidence >= 0.6)

        if fused.score >= self.orch.soft_block or hard:
            demand = RED
        elif fused.score >= self.orch.soft_alert or single_esc:
            demand = YELLOW
        else:
            demand = GREEN

        # 3) 迟滞状态机
        prev = self.state
        if self.state == GREEN:
            if demand == RED:
                self.state = RED
                self._clean_streak = 0
            elif demand == YELLOW:
                self._alert_streak += 1
                if self._alert_streak >= self.yellow_windows:
                    self.state = YELLOW
            else:
                self._alert_streak = 0
        elif self.state == YELLOW:
            if demand == RED:
                self.state = RED
                self._clean_streak = 0
            elif demand == YELLOW:
                self._clean_streak = 0
            else:
                self._clean_streak += 1
                if self._clean_streak >= self.recover_windows:
                    self.state = GREEN
        else:  # RED
            if demand == RED:
                self._clean_streak = 0
            else:
                self._clean_streak += 1
                if self._clean_streak >= self.recover_windows:
                    self.state = GREEN

        if any(c.confidence == 0.0 for c in (acoustic, voiceprint, semantic)):
            self.offline = True

        # 4) 组装事件
        if semantic.detail and "转写:" in semantic.detail:
            self.last_transcript = semantic.detail.split("转写:", 1)[1][:60]

        events: list[dict] = [{
            "type": "window",
            "idx": idx,
            "t_rel": round(t_rel, 2),
            "channels": {
                "acoustic": {"score": sac.score, "label": sac.label},
                "voiceprint": {"score": svp.score, "label": svp.label},
                "semantic": {"score": ssem.score, "label": ssem.label,
                             "transcript": self.last_transcript},
            },
            "fused": fused.score,
            "state": self.state,
            "offline": self.offline,
        }]

        # 5) 进入 RED 时发告警（受冷却约束）。
        #    新进入 RED，或持续 RED 但已度过冷却期，都发一次告警——
        #    既避免每窗轰炸老人，又保证持续诈骗中每 cooldown 秒提醒一次。
        if self.state == RED and t_rel - self._last_alert_t >= self.cooldown:
            self._last_alert_t = t_rel
            events.append({
                "type": "alert",
                "state": RED,
                "rationale": fused.rationale,
                "advice": ADVICE.format(家人=self.family_name),
            })
        return events

    # ------------------------------------------------------------------ #
    @property
    def current_state(self) -> str:
        return self.state
