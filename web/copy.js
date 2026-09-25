/* 谛听 VeriCall · 文案库（P2-3 适老交互）
 * ------------------------------------------------------------------
 * 非指控式原则：不指责老人、不恐吓，只给一个可执行的下一步动作。
 * 三处共用：实时告警弹窗 / 语音播报（SpeechSynthesis）/ 子女分享卡片。
 * 由 FastAPI StaticFiles 以 /copy.js 提供；index.html 内有降级兜底。
 */
window.VERICALL_COPY = {
  /* state: "green" | "yellow" | "red"
   * word  —— 大字标题（整屏色块 / 告警卡）
   * short —— 副标（一句话动作）
   * advise—— 播报与卡片用的完整劝慰句，{family} 会被替换成已登记家人称谓 */
  states: {
    green: {
      word: "通话正常",
      short: "放心聊",
      advise: "通话正常，放心聊。"
    },
    yellow: {
      word: "话术可疑",
      short: "多留个心眼",
      advise: "这通电话提到转账，请多留个心眼，和家里人商量一下再决定。"
    },
    red: {
      word: "请先核实",
      short: "先挂断，回拨确认",
      advise: "这个电话的声音有些不对劲，建议先挂断，拨打{family}平时的号码确认一下。"
    }
  },
  /* 子女分享卡片文案 */
  card: {
    title: "谛听拦截提醒",
    subtitle: "AI 拟声电话诈骗 · 三通道联合判定",
    action: "建议这样做",
    footer: "谛听 VeriCall · 声学伪造 × 声纹核验 × 话术理解",
    hint: "长按保存图片，转发给子女或家人群"
  },
  /* 完整劝慰句（播报 / 卡片通用），family 缺省为「家人」 */
  advise: function (state, family) {
    var st = this.states[state] || this.states.green;
    return st.advise.replace("{family}", family || "家人");
  },
  /* 语音播报整句（与 advise 同句，方便统一调整语气） */
  speak: function (state, family) {
    return this.advise(state, family);
  }
};
