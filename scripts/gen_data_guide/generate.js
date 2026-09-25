// 谛听 VeriCall 数据获取方式详册 — docx 生成脚本
// 设计：R1 封面（Pure Paragraph Left）+ DM-1 Deep Cyan 色板（科技/AI）
// 结构：3 节（封面无页码 / 前置目录罗马数字 / 正文阿拉伯数字从 1 起）
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, PageNumber, NumberFormat, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, ShadingType, SectionType, TableOfContents,
  PageBreak, LevelFormat, TableLayoutType,
} = require("docx");
const fs = require("fs");

// ── DM-1 色板 ──
const PAL = {
  bg: "162235", accent: "37DCF2",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "1B6B7A", headerText: "FFFFFF", accentLine: "1B6B7A", innerLine: "C8DDE2", surface: "EDF3F5" },
  headingColor: "0E2A33", bodyColor: "1A2430", secondary: "506070",
  ok: "2A7A50", warn: "B45309",
};

const FONT_BODY = { ascii: "Calibri", eastAsia: "Microsoft YaHei" };
const FONT_HEAD = { ascii: "Calibri", eastAsia: "SimHei" };
const FONT_MONO = { ascii: "Consolas", eastAsia: "Microsoft YaHei" };

// ── 封面辅助（design-system.md 官方配方）──
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
const allNoBorders = { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB };

function splitTitleLines(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const breakAfter = new Set([..."\uFF0C\u3002\u3001\uFF1B\uFF1A\uFF01\uFF1F", ..."\u7684\u4E0E\u548C\u53CA\u4E4B\u5728\u4E8E\u4E3A", ..."-_\u2014\u2013\u00B7/", ..." \t"]);
  const lines = [];
  let remaining = title;
  while (remaining.length > charsPerLine) {
    let breakAt = -1;
    for (let i = charsPerLine; i >= Math.floor(charsPerLine * 0.6); i--) {
      if (i < remaining.length && breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    }
    if (breakAt === -1) {
      const limit = Math.min(remaining.length, Math.ceil(charsPerLine * 1.3));
      for (let i = charsPerLine + 1; i < limit; i++) {
        if (breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
      }
    }
    if (breakAt === -1) {
      breakAt = charsPerLine;
      const prevChar = remaining[breakAt - 1], nextChar = remaining[breakAt];
      if (prevChar && nextChar && !breakAfter.has(prevChar) && !breakAfter.has(nextChar) &&
          /[\u4e00-\u9fff]/.test(prevChar) && /[\u4e00-\u9fff]/.test(nextChar)) breakAt -= 1;
    }
    lines.push(remaining.slice(0, breakAt).trim());
    remaining = remaining.slice(breakAt).trim();
  }
  if (remaining) lines.push(remaining);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop();
    lines[lines.length - 1] += last;
  }
  return lines;
}

function calcTitleLayout(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  const charWidth = (pt) => pt * 20;
  const charsPerLine = (pt) => Math.floor(maxWidthTwips / charWidth(pt));
  let titlePt = preferredPt, lines;
  while (titlePt >= minPt) {
    const cpl = charsPerLine(titlePt);
    if (cpl < 2) { titlePt -= 2; continue; }
    lines = splitTitleLines(title, cpl);
    if (lines.length <= 3) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    lines = splitTitleLines(title, charsPerLine(minPt));
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}

function calcCoverSpacing(params) {
  const { titleLineCount = 1, titlePt = 36, hasSubtitle = false, hasEnglishLabel = false,
    metaLineCount = 0, fixedHeight = 800, pageHeight = 16838, marginTop = 0, marginBottom = 0 } = params;
  const SAFETY = 1200;
  const usableHeight = pageHeight - marginTop - marginBottom - SAFETY;
  const titleHeight = titleLineCount * (titlePt * 23 + 200);
  const subtitleHeight = hasSubtitle ? (12 * 23 + 600) : 0;
  const englishLabelHeight = hasEnglishLabel ? (9 * 23 + 600) : 0;
  const metaHeight = metaLineCount * (10 * 23 + 100);
  const implicitParaHeight = 3 * 300;
  const contentHeight = titleHeight + subtitleHeight + englishLabelHeight + metaHeight + fixedHeight + implicitParaHeight;
  const safeRemaining = Math.max(usableHeight - contentHeight, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(safeRemaining * 0.45), rawBottom = Math.floor(safeRemaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  return { topSpacing, midSpacing: Math.max(safeRemaining - topSpacing - bottomSpacing, 0), bottomSpacing };
}

function buildCoverR1(config) {
  const P = config.palette;
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length, fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: P.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));
  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: P.accent, space: 8 } },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "), size: 18, color: P.accent, font: { ascii: "Calibri", eastAsia: "SimHei" }, characterSpacing: 40 })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true, color: P.titleColor, font: { eastAsia: "SimHei", ascii: "Arial" } })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL }, spacing: { after: 800 },
      children: [new TextRun({ text: config.subtitle, size: 24, color: P.subtitleColor, font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      indent: { left: padL + 200 }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({ text: line, size: 24, color: P.metaColor, font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));
  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: P.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: config.footerLeft || "", size: 16, color: P.footerColor, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" } }),
      new TextRun({ text: "                                        " }),
      new TextRun({ text: config.footerRight || "", size: 16, color: P.footerColor, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" } }),
    ],
  }));
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({ shading: { type: ShadingType.CLEAR, fill: P.bg }, borders: noBorders, children })],
    })],
  })];
}

// ── 正文组件 ──
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160, line: 312 },
    children: [new TextRun({ text, bold: true, size: 32, color: PAL.headingColor, font: FONT_HEAD })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120, line: 312 },
    children: [new TextRun({ text, bold: true, size: 28, color: PAL.headingColor, font: FONT_HEAD })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100, line: 312 },
    children: [new TextRun({ text, bold: true, size: 24, color: PAL.headingColor, font: FONT_HEAD })],
  });
}
function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 480 }, spacing: { line: 312, after: 60 },
    children: [new TextRun({ text, size: 24, color: PAL.bodyColor, font: FONT_BODY, ...opts })],
  });
}
// 带内联着色片段的正文段（用于 核实状态 标记）
function bodyRich(runs) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 480 }, spacing: { line: 312, after: 60 },
    children: runs.map(r => new TextRun({ size: 24, color: PAL.bodyColor, font: FONT_BODY, ...r })),
  });
}
function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 }, spacing: { line: 312, after: 40 },
    children: [new TextRun({ text, size: 24, color: PAL.bodyColor, font: FONT_BODY })],
  });
}
const numberingConfigs = [];
let numSeq = 0;
function numberedList(items) {
  const ref = "num-list-" + (++numSeq);
  numberingConfigs.push({
    reference: ref,
    levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
  });
  return items.map(t => new Paragraph({
    numbering: { reference: ref, level: 0 }, spacing: { line: 312, after: 40 },
    children: [new TextRun({ text: t, size: 24, color: PAL.bodyColor, font: FONT_BODY })],
  }));
}
function codeLine(text) {
  return new Paragraph({
    spacing: { line: 276, before: 0, after: 0 },
    shading: { type: ShadingType.CLEAR, fill: PAL.table.surface },
    indent: { left: 360, right: 360 },
    children: [new TextRun({ text, size: 18, color: PAL.headingColor, font: FONT_MONO })],
  });
}
function codeBlock(lines) {
  const out = [new Paragraph({ spacing: { before: 60, after: 0 }, children: [] })];
  for (const ln of lines) out.push(codeLine(ln));
  out.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
  return out;
}
// 提示/警示框（左侧强调竖线）
function noteBox(text, kind = "info") {
  const color = kind === "warn" ? PAL.warn : PAL.table.accentLine;
  return new Paragraph({
    spacing: { before: 100, after: 100, line: 312 },
    indent: { left: 360, right: 360 },
    shading: { type: ShadingType.CLEAR, fill: PAL.table.surface },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color, space: 10 } },
    children: [new TextRun({ text, size: 21, color: kind === "warn" ? PAL.warn : PAL.headingColor, font: FONT_BODY })],
  });
}
const cellMargins = { top: 60, bottom: 60, left: 120, right: 120 };
function th(text, widthPct) {
  return new TableCell({
    shading: { type: ShadingType.CLEAR, fill: PAL.table.headerBg }, margins: cellMargins,
    width: widthPct ? { size: widthPct, type: WidthType.PERCENTAGE } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, size: 21, color: PAL.table.headerText, font: FONT_BODY })] })],
  });
}
function td(content, widthPct, opts = {}) {
  const runs = Array.isArray(content) ? content.map(r => new TextRun({ size: 21, color: PAL.bodyColor, font: FONT_BODY, ...r })) : [new TextRun({ text: content, size: 21, color: PAL.bodyColor, font: FONT_BODY, ...opts })];
  return new TableCell({
    margins: cellMargins, width: widthPct ? { size: widthPct, type: WidthType.PERCENTAGE } : undefined,
    shading: opts.fill ? { type: ShadingType.CLEAR, fill: opts.fill } : undefined,
    children: [new Paragraph({ children: runs })],
  });
}
function dataTable(headers, rows, widths, caption) {
  const out = [];
  if (caption) out.push(new Paragraph({
    keepNext: true, spacing: { before: 160, after: 80 },
    children: [new TextRun({ text: caption, bold: true, size: 21, color: PAL.headingColor, font: FONT_HEAD })],
  }));
  out.push(new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: PAL.table.accentLine },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: PAL.table.accentLine },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: PAL.table.innerLine },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((t, i) => th(t, widths ? widths[i] : undefined)) }),
      ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((c, i) => (c && c.__cell) ? c.node : td(c, widths ? widths[i] : undefined)) })),
    ],
  }));
  return out;
}
// 状态标记辅助
const OK = { text: "\u2705 2026-09-03 \u5DF2\u6838\u5B9E", color: PAL.ok, bold: true };   // ✅ 已核实
const WARN = { text: "\u26A0\uFE0F \u672A\u80FD\u5728\u7EBF\u6838\u5B9E\uFF0C\u5F85\u590D\u6838", color: PAL.warn, bold: true }; // ⚠️ 待复核
function cellStatus(verified) { return { __cell: true, node: td([verified ? OK : WARN]) }; }

// ── 正文内容 ──
const bodyChildren = [];

// 第 1 章 文档说明与核实状态
bodyChildren.push(h1("\u7B2C 1 \u7AE0  \u6587\u6863\u8BF4\u660E\u4E0E\u94FE\u63A5\u6838\u5B9E\u72B6\u6001"));
bodyChildren.push(h2("1.1 \u7F16\u5236\u76EE\u7684"));
bodyChildren.push(body("\u8C1B\u542C VeriCall \u662F\u9762\u5411\u5BB6\u5EAD\u573A\u666F\u7684 AI \u8BED\u97F3\u5192\u5145\u8BC8\u9A97\u62E6\u622A\u7CFB\u7EDF\uFF0C\u91C7\u7528\u4E09\u901A\u9053\u878D\u5408\u67B6\u6784\uFF1A\u58F0\u5B66\u9632\u4F2A\uFF08AASIST \u539F\u59CB\u6CE2\u5F62\u53CD\u6B3A\u9A97\uFF09\u3001\u5BB6\u5EAD\u58F0\u7EB9\u6838\u9A8C\uFF08CAMPPlus \u8BF4\u8BDD\u4EBA\u5D4C\u5165\u6BD4\u5BF9\uFF09\u4E0E\u8BC8\u9A97\u8BDD\u672F\u8BED\u4E49\u5206\u6790\uFF08SenseVoice \u8F6C\u5199 + deepseek-r1:8b \u5927\u6A21\u578B\u63A8\u7406\uFF09\u3002\u4E09\u4E2A\u901A\u9053\u7684\u8BAD\u7EC3\u3001\u5FAE\u8C03\u4E0E\u8BC4\u4F30\u5747\u4F9D\u8D56\u5927\u91CF\u3001\u9AD8\u8D28\u91CF\u4E14\u6765\u6E90\u5408\u89C4\u7684\u6570\u636E\u3002\u672C\u518C\u7ED9\u51FA\u6BCF\u4E2A\u6570\u636E\u96C6\u7684\u83B7\u53D6\u5165\u53E3\u3001\u5177\u4F53\u6B65\u9AA4\u3001\u8BB8\u53EF\u8BC1\u7EA6\u675F\u53CA\u5176\u5728\u672C\u9879\u76EE\u4E2D\u7684\u7528\u9014\uFF0C\u4F9B\u56E2\u961F\u6210\u5458\u6309\u7AE0\u6267\u884C\u3002"));
bodyChildren.push(body("\u672C\u518C\u6240\u6709\u5728\u7EBF\u94FE\u63A5\u5747\u4E8E 2026-09-03 \u901A\u8FC7\u81EA\u52A8\u5316\u6293\u53D6\u9010\u4E00\u6838\u5B9E\uFF08\u76F4\u63A5\u62C9\u53D6\u5B98\u65B9 README \u6216\u6570\u636E\u9875\u539F\u6587\u6BD4\u5BF9\uFF09\u3002\u6838\u5B9E\u901A\u8FC7\u7684\u6807\u6CE8\u4E3A\u2705\uFF1B\u56E0\u7F51\u7EDC\u4E0D\u53EF\u8FBE\u6216\u5B98\u65B9\u5165\u53E3\u53D8\u66F4\u800C\u672A\u80FD\u6838\u5B9E\u7684\uFF0C\u660E\u786E\u6807\u6CE8\u4E3A\u26A0\uFE0F \u5E76\u7ED9\u51FA\u590D\u6838\u6307\u5F15\uFF0C\u4E0D\u63D0\u4F9B\u672A\u7ECF\u6838\u5B9E\u7684\u4E0B\u8F7D\u5730\u5740\uFF0C\u4EE5\u514D\u8BEF\u5BFC\u3002\u5F15\u7528\u672C\u518C\u65F6\u8BF7\u4EE5\u5404\u6570\u636E\u96C6\u5B98\u65B9\u9875\u9762\u4E3A\u51C6\u3002"));
bodyChildren.push(h2("1.2 \u6570\u636E\u96C6\u6838\u5B9E\u72B6\u6001\u603B\u8868"));
bodyChildren.push(...dataTable(
  ["\u6570\u636E\u96C6", "\u7C7B\u578B", "\u6838\u5B9E\u72B6\u6001", "\u83B7\u53D6\u5165\u53E3"],
  [
    ["MUSAN", "\u566A\u58F0\u589E\u5F3A\u8BED\u6599", cellStatus(true), "openslr.org/17"],
    ["AISHELL-3", "\u4E2D\u6587\u591A\u8BF4\u8BDD\u4EBA", cellStatus(true), "openslr.org/93"],
    ["CFAD", "\u4E2D\u6587\u5BF9\u6297\u9632\u4F2A", cellStatus(true), "github.com/ADDchallenge/CFAD"],
    ["FMFCC-A", "\u4E2D\u6587\u9632\u4F2A", cellStatus(true), "github.com/Amforever/FMFCC-A"],
    ["WaveFake", "\u82F1\u6587\u9632\u4F2A", cellStatus(true), "github.com/RUB-SysSec/WaveFake"],
    ["GPT-SoVITS", "\u7EA2\u961F\u514B\u9686\u5DE5\u5177", cellStatus(true), "github.com/RVC-Boss/GPT-SoVITS"],
    ["CosyVoice", "\u7EA2\u961F\u514B\u9686\u5DE5\u5177", cellStatus(true), "github.com/FunAudioLLM/CosyVoice"],
    ["Seed-VC", "\u7EA2\u961F\u514B\u9686\u5DE5\u5177", cellStatus(true), "github.com/Plachtaa/seed-vc"],
    ["ASVspoof \u7CFB\u5217", "\u82F1\u6587\u9632\u4F2A\u57FA\u51C6", cellStatus(false), "asvspoof.org\uFF08\u5B98\u7F51\u6CE8\u518C\uFF09"],
    ["MLAAD", "\u591A\u8BED\u79CD\u9632\u4F2A", cellStatus(false), "\u8BBA\u6587\u9875 / HuggingFace \u68C0\u7D22"],
    ["In-the-Wild", "\u8DE8\u57DF\u8BC4\u4F30", cellStatus(false), "asvspoof.org/in-the-wild"],
    ["PartialSpoof", "\u90E8\u5206\u4F2A\u9020\u8BC4\u4F30", cellStatus(false), "\u8BBA\u6587\u4F5C\u8005\u9875\u9762"],
  ],
  [24, 24, 28, 24],
  "\u8868 1-1  \u6570\u636E\u96C6\u6838\u5B9E\u72B6\u6001\uFF08\u6838\u5B9E\u65E5\u671F 2026-09-03\uFF09"
));
bodyChildren.push(noteBox("\u5F85\u590D\u6838\u9879\u8BF4\u660E\uFF1A\u672C\u6B21\u6838\u5B9E\u65F6 asvspoof.org \u53CA\u5176\u5B50\u9875\u9762\u4ECE\u5F53\u524D\u7F51\u7EDC\u8FDE\u63A5\u91CD\u7F6E\u4E0D\u53EF\u8FBE\uFF08\u5C5E\u7F51\u7EDC\u95EE\u9898\u800C\u975E\u94FE\u63A5\u5931\u6548\uFF09\uFF1B\u5904\u7406\u529E\u6CD5\u89C1\u5404\u5BF9\u5E94\u5C0F\u8282\u7684\u590D\u6838\u6307\u5F15\u3002", "warn"));

// 第 2 章 数据需求总览
bodyChildren.push(h1("\u7B2C 2 \u7AE0  \u6570\u636E\u9700\u6C42\u603B\u89C8"));
bodyChildren.push(h2("2.1 \u4E09\u901A\u9053\u4E0E\u6570\u636E\u96C6\u6620\u5C04"));
bodyChildren.push(body("\u4E09\u901A\u9053\u5BF9\u6570\u636E\u7684\u9700\u6C42\u5404\u4E0D\u76F8\u540C\uFF1A\u58F0\u5B66\u9632\u4F2A\u901A\u9053\u9700\u8981\u771F/\u4F2A\u4E8C\u5206\u6807\u6CE8\u7684\u5927\u89C4\u6A21\u8BED\u97F3\uFF1B\u58F0\u7EB9\u6838\u9A8C\u901A\u9053\u9700\u8981\u4E2D\u6587\u591A\u8BF4\u8BDD\u4EBA\u8BED\u6599\u4E0E\u566A\u58F0\u589E\u5F3A\u6750\u6599\uFF1B\u8BED\u4E49\u901A\u9053\u9700\u8981\u8BC8\u9A97\u8BDD\u672F\u6587\u672C\uFF1B\u7EA2\u961F\u8BC4\u4F30\u5219\u9700\u8981\u81EA\u5236\u6DF1\u5EA6\u4F2A\u9020\u8BED\u97F3\u3002\u6620\u5C04\u5173\u7CFB\u5982\u8868 2-1\u3002"));
bodyChildren.push(...dataTable(
  ["\u901A\u9053", "\u6A21\u578B", "\u6570\u636E\u9700\u6C42", "\u5BF9\u5E94\u6570\u636E\u96C6"],
  [
    ["\u58F0\u5B66\u9632\u4F2A", "AASIST", "\u771F/\u4F2A\u6807\u6CE8\u8BED\u97F3\uFF08\u8BAD\u7EC3+\u8BC4\u4F30\uFF09", "ASVspoof 2019/2021/5\u3001WaveFake\u3001FMFCC-A\u3001CFAD\u3001MLAAD\u3001In-the-Wild\u3001PartialSpoof"],
    ["\u58F0\u7EB9\u6838\u9A8C", "CAMPPlus", "\u4E2D\u6587\u591A\u8BF4\u8BDD\u4EBA\u6CE8\u518C\u8BED\u6599 + \u566A\u58F0\u589E\u5F3A", "AISHELL-3\u3001MUSAN\u3001\u5BB6\u5EAD\u58F0\u7EB9\u81EA\u91C7"],
    ["\u8BED\u4E49\u5206\u6790", "SenseVoice + deepseek-r1:8b", "\u8BC8\u9A97\u8BDD\u672F\u6587\u672C\uFF08\u5206\u7C7B\u6807\u6CE8\uFF09", "\u81EA\u5EFA\u8BDD\u672F\u8BED\u6599\uFF08\u516C\u5F00\u901A\u62A5 + LLM \u6269\u589E\uFF09"],
    ["\u7EA2\u961F\u8BC4\u4F30", "\u5168\u901A\u9053", "\u5BF9\u6297\u6027\u6DF1\u5EA6\u4F2A\u9020\u8BED\u97F3", "GPT-SoVITS\u3001CosyVoice\u3001Seed-VC \u81EA\u5236"],
  ],
  [14, 22, 28, 36],
  "\u8868 2-1  \u4E09\u901A\u9053\u4E0E\u6570\u636E\u96C6\u6620\u5C04"
));
bodyChildren.push(h2("2.2 \u5B58\u50A8\u89C4\u5212\u4E0E\u76EE\u5F55\u5E03\u5C40"));
bodyChildren.push(body("\u6240\u6709\u6570\u636E\u7EDF\u4E00\u5B58\u653E\u4E8E\u72EC\u7ACB\u6570\u636E\u76D8 D:/VeriCall_data\uFF08\u4E0E\u4EE3\u7801\u4ED3\u5E93\u5206\u79BB\uFF0C\u4FBF\u4E8E\u5907\u4EFD\u4E0E\u7248\u672C\u7BA1\u7406\uFF09\u3002\u76EE\u5F55\u6309\u6570\u636E\u96C6\u5206\u76EE\uFF0C\u539F\u59CB\u5305\u4E0E\u89E3\u538B\u540E\u6570\u636E\u5206\u5F00\u5B58\u653E\uFF0C\u7EA2\u961F\u5BF9\u6297\u96C6\u4E0E\u8BAD\u7EC3\u96C6\u7269\u7406\u9694\u79BB\uFF1A"));
bodyChildren.push(...codeBlock([
  "D:/VeriCall_data/",
  "\u251C\u2500 archives/                  # \u539F\u59CB\u4E0B\u8F7D\u5305\uFF08tar.gz / zip\uFF09",
  "\u251C\u2500 asvspoof2019_LA/          # \u58F0\u5B66\u9632\u4F2A\u4E3B\u529B\u8BAD\u7EC3\u96C6",
  "\u251C\u2500 asvspoof2021_LA/  asvspoof5/",
  "\u251C\u2500 wavefake/  fmfcc_a/  cfad/  mlaad/",
  "\u251C\u2500 in_the_wild/  partialspoof/  # \u8DE8\u57DF\u8BC4\u4F30\u4E13\u7528\uFF08\u7981\u5165\u8BAD\u7EC3\u7BA1\u7EBF\uFF09",
  "\u251C\u2500 aishell3/  musan/           # \u58F0\u7EB9\u901A\u9053",
  "\u251C\u2500 family_voiceprints/         # \u5BB6\u5EAD\u81EA\u91C7\u58F0\u7EB9\uFF08\u52A0\u5BC6\u5B58\u50A8\uFF09",
  "\u251C\u2500 scam_scripts/               # \u8BDD\u672F\u8BED\u6599 JSONL",
  "\u2514\u2500 redteam/                    # \u7EA2\u961F\u81EA\u5236\u4F2A\u9020\u8BED\u97F3\uFF08\u7981\u5165\u8BAD\u7EC3\u7BA1\u7EBF\uFF09",
]));
bodyChildren.push(body("\u5DF2\u6838\u5B9E\u4F53\u79EF\uFF1AMUSAN \u7EA6 11 GB\u3001AISHELL-3 \u7EA6 19 GB\uFF1B\u5176\u4F59\u6570\u636E\u96C6\u4EE5\u5B98\u65B9\u53D1\u5E03\u4E3A\u51C6\uFF0C\u5EFA\u8BAE\u9884\u7559\u4E0D\u4F4E\u4E8E 300 GB \u7A7A\u95F2\u7A7A\u95F4\uFF08\u542B\u89E3\u538B\u4E0E\u4E2D\u95F4\u4EA7\u7269\uFF09\u3002"));

// 第 3 章 声学防伪通道
bodyChildren.push(h1("\u7B2C 3 \u7AE0  \u58F0\u5B66\u9632\u4F2A\u901A\u9053\u6570\u636E\u96C6"));
bodyChildren.push(h2("3.1 ASVspoof 2019 / 2021 / 5\uFF08\u82F1\u6587\u9632\u4F2A\u57FA\u51C6\uFF0C\u5B98\u7F51\u6CE8\u518C\u5236\uFF09"));
bodyChildren.push(bodyRich([
  { text: "\u7B80\u4ECB\uFF1A" },
  { text: "ASVspoof \u662F\u58F0\u7EB9\u53CD\u6B3A\u9A97\u9886\u57DF\u6700\u6743\u5A01\u7684\u7CFB\u5217\u57FA\u51C6\uFF0C\u81EA 2015 \u5E74\u8D77\u6BCF\u4E24\u5E74\u4E00\u5C4A\u3002\u672C\u9879\u76EE AASIST \u57FA\u7EBF\u5373\u5728 ASVspoof 2019 LA \u5B50\u96C6\u4E0A\u590D\u73B0\uFF08\u9879\u76EE\u5185\u5B9E\u6D4B dev EER 0.745% / eval EER 3.49%\uFF09\u3002", }
]));
bodyChildren.push(...numberedList([
  "\u8BBF\u95EE\u5B98\u7F51 asvspoof.org\uFF0C\u8FDB\u5165\u5BF9\u5E94\u5C4A\u6B21\uFF08ASVspoof 2019 / 2021 / 5\uFF09\u7684\u6570\u636E\u9875\u9762\u3002",
  "\u6CE8\u518C\u8D26\u53F7\u5E76\u7B7E\u7F72\u6700\u7EC8\u7528\u6237\u8BB8\u53EF\u534F\u8BAE\uFF08EULA\uFF0C\u4EC5\u9650\u5B66\u672F\u7814\u7A76\u7528\u9014\uFF09\u3002",
  "\u7B7E\u7F72\u540E\u5728\u4E2A\u4EBA\u4E2D\u5FC3\u83B7\u53D6\u4E0B\u8F7D\u94FE\u63A5\u6216\u51ED\u8BC1\uFF0C\u4E0B\u8F7D LA\uFF08\u903B\u8F91\u8BBF\u95EE\uFF09\u5B50\u96C6\u3002",
  "\u4F18\u5148\u4E0B\u8F7D\uFF1a2019 LA\uFF08\u57FA\u7EBF\u8BAD\u7EC3/\u8BC4\u4F30\uFF09\u2192 2021 LA\uFF08\u538B\u7F29\u4FE1\u9053\u6CDB\u5316\uFF09\u2192 ASVspoof 5\uFF08\u591A\u8BED\u79CD\u4F17\u5305\u4E0E\u65B0\u578B\u653B\u51FB\uFF0C\u89C4\u6A21\u4EE5\u5B98\u7F51\u4E3A\u51C6\uFF09\u3002",
  "\u89E3\u538B\u540E\u6309\u7B2C 7 \u7AE0\u89C4\u8303\u7EDF\u4E00\u8F6C 16 kHz \u5355\u58F0\u9053\u5E76\u751F\u6210 meta \u6E05\u5355\u3002",
]));
bodyChildren.push(noteBox("\u590D\u6838\u6307\u5F15\uFF1A2026-09-03 \u5B98\u7F51\u4ECE\u672C\u5730\u7F51\u7EDC\u4E0D\u53EF\u8FBE\uFF08\u8FDE\u63A5\u91CD\u7F6E\uFF09\u3002\u8BF7\u5728\u6821\u56ED\u7F51/\u624B\u673A\u70ED\u70B9\u4E0B\u91CD\u8BD5\uFF0C\u6216\u901A\u8FC7\u8BBA\u6587 Dataset \u9875\u811A\u6CE8\u7684\u5B98\u65B9\u94FE\u63A5\u5165\u53E3\u8FDB\u5165\uFF1B\u5B98\u7F51\u6CE8\u518C\u5236\u591A\u5E74\u7A33\u5B9A\uFF0C\u6D41\u7A0B\u4E0D\u53D8\u3002", "warn"));

bodyChildren.push(h2("3.2 WaveFake\uFF08\u82F1\u6587 GAN \u5408\u6210\u8BED\u97F3\uFF09"));
bodyChildren.push(body("\u7B80\u4ECB\uFF1ANeurIPS 2021 Datasets & Benchmarks \u6536\u5F55\uFF0C\u542B\u7EA6 11.8 \u4E07\u6761 GAN \u7C7B\u5408\u6210\u8BED\u97F3\uFF08\u591A\u79CD GAN \u58F0\u7801\u5668/\u8F6C\u6362\u7CFB\u7EDF\uFF0C\u4EE5 README \u4E3A\u51C6\uFF09\uFF0C\u4E0E\u771F\u5B9E\u8BED\u97F3\u642D\u914D\u4F7F\u7528\u3002\u672C\u9879\u76EE\u7528\u4F5C\u82F1\u6587\u8DE8\u57DF\u6CDB\u5316\u8BC4\u4F30\u3002\u5165\u53E3 github.com/RUB-SysSec/WaveFake\uFF08\u5DF2\u6838\u5B9E\uFF09\uFF0CREADME \u63D0\u4F9B Google Drive / Zenodo \u6253\u5305\u4E0B\u8F7D\u4E0E\u6BCF\u4E2A\u751F\u6210\u7CFB\u7EDF\u7684\u76EE\u5F55\u7ED3\u6784\u8BF4\u660E\u3002"));
bodyChildren.push(...numberedList([
  "git clone https://github.com/RUB-SysSec/WaveFake.git",
  "\u9605\u8BFB README \u4E2D Download \u5C0F\u8282\uFF0C\u4ECE Google Drive \u6216 Zenodo \u4E0B\u8F7D generated_audio \u6253\u5305\u6587\u4EF6\u3002",
  "\u6821\u9A8C\u5404\u751F\u6210\u7CFB\u7EDF\u5B50\u76EE\u5F55\u4E0E README \u6E05\u5355\u4E00\u81F4\u540E\u518D\u5165\u5E93\u3002",
]));

bodyChildren.push(h2("3.3 FMFCC-A\uFF08\u4E2D\u6587\u9632\u4F2A\u4E3B\u529B\uFF09"));
bodyChildren.push(body("\u7B80\u4ECB\uFF1AFMFCC \u7ADE\u8D5B\u97F3\u9891\u8D5B\u9053\u6570\u636E\u96C6\uFF0C\u4E2D\u6587 16 kHz wav\uFF0C\u8BAD\u7EC3\u7EA6 1 \u4E07\u6761 / \u8BC4\u4F30\u7EA6 4 \u4E07\u6761\uFF08\u4EE5 README \u4E3A\u51C6\uFF09\uFF0C\u662F\u672C\u9879\u76EE\u58F0\u5B66\u901A\u9053\u4E2D\u6587\u57DF\u5FAE\u8C03\u7684\u4E3B\u529B\u3002\u5165\u53E3 github.com/Amforever/FMFCC-A\uFF08\u5DF2\u6838\u5B9E\uFF0CREADME \u63D0\u4F9B\u767E\u5EA6\u7F51\u76D8\u5206\u4EAB\u94FE\u63A5\u4E0E\u63D0\u53D6\u7801\uFF09\u3002"));
bodyChildren.push(...numberedList([
  "\u6253\u5F00 github.com/Amforever/FMFCC-A \u7684 README\uFF0C\u83B7\u53D6\u767E\u5EA6\u7F51\u76D8\u94FE\u63A5\u4E0E\u63D0\u53D6\u7801\uFF08\u63D0\u53D6\u7801\u4EE5 README \u5F53\u524D\u7248\u672C\u4E3A\u51C6\uFF09\u3002",
  "\u4E0B\u8F7D\u540E\u6821\u9A8C\u6587\u4EF6\u6570\u4E0E README \u8BB0\u8F7D\u4E00\u81F4\uFF0C\u89E3\u538B\u81F3 D:/VeriCall_data/fmfcc_a/\u3002",
  "\u6309 README \u8BF4\u660E\u533A\u5206 train/eval \u4E0E\u771F/\u4F2A\u6807\u7B7E\uFF0C\u751F\u6210\u7EDF\u4E00 meta.csv\u3002",
]));

bodyChildren.push(h2("3.4 CFAD\uFF08\u4E2D\u6587\u6301\u7EED\u5BF9\u6297\u66F4\u65B0\uFF09"));
bodyChildren.push(body("\u7B80\u4ECB\uFF1A\u9762\u5411\u5B9E\u9645\u5E94\u7528\u7684\u4E2D\u6587\u9632\u4F2A\u6570\u636E\u96C6\uFF0C\u5206\u9636\u6BB5\u6301\u7EED\u6295\u653E\u65B0\u578B\u4F2A\u9020\u6837\u672C\uFF08\u5BF9\u6297\u6027\u66F4\u65B0\u8D5B\u5236\uFF09\uFF0C\u89C4\u6A21\u7EA6 3.86 \u4E07 / 7.72 \u4E07\u6761\uFF08\u4EE5\u5B98\u65B9\u9875\u9762\u4E3A\u51C6\uFF09\u3002\u672C\u9879\u76EE\u7528\u4F5C\u5BF9\u6297\u6F02\u79FB\u573A\u666F\u7684\u6301\u7EED\u8BC4\u4F30\u3002"));
bodyChildren.push(...numberedList([
  "\u6253\u5F00\u4ED3\u5E93 github.com/ADDchallenge/CFAD\uFF08\u5DF2\u6838\u5B9E\uFF09\uFF0C\u9605\u8BFB\u5F53\u524D\u8F6E\u6B21\u7684\u6570\u636E\u7533\u8BF7\u6D41\u7A0B\u3002",
  "\u6570\u636E\u4E3B\u4F53\u6258\u7BA1\u4E8E Zenodo\uFF08\u5DF2\u6838\u5B9E\u8BB0\u5F55\u9875 zenodo.org/records/8122764\uFF0C\u540E\u7EED\u8F6E\u6B21\u4EE5\u4ED3\u5E93\u94FE\u63A5\u4E3A\u51C6\uFF09\uFF0C\u6309\u9875\u9762\u6307\u5F15\u4E0B\u8F7D\u3002",
  "\u5B66\u672F\u80CC\u666F\u53C2\u8003\u8BBA\u6587 arXiv 2207.12308\u3002",
]));

bodyChildren.push(h2("3.5 MLAAD\uFF08\u591A\u8BED\u79CD\u9632\u4F2A\uFF09"));
bodyChildren.push(body("\u7B80\u4ECB\uFF1A\u591A\u8BED\u79CD\u97F3\u9891\u53CD\u6B3A\u9A97\u6570\u636E\u96C6\uFF08\u9879\u76EE\u5185\u6587\u6863\u8BB0\u8F7D\u7EA6 7.6 \u4E07\u6761\u3001\u8986\u76D6\u7EA6 23 \u79CD\u8BED\u8A00\u4E0E\u6570\u5341\u4E2A TTS \u7CFB\u7EDF\uFF0C\u5747\u4EE5\u8BBA\u6587\u4E0E\u5B98\u65B9\u9875\u9762\u4E3A\u51C6\uFF09\u3002\u672C\u9879\u76EE\u4EC5\u8BA1\u5212\u53D6\u5176\u4E2D\u6587\u5B50\u96C6\u505A\u8DE8\u7CFB\u7EDF\u6CDB\u5316\u9A8C\u8BC1\uFF0C\u4F18\u5148\u7EA7\u4F4E\u4E8E FMFCC-A \u4E0E CFAD\u3002"));
bodyChildren.push(...numberedList([
  "\u68C0\u7D22\u8BBA\u6587\u300AMLAAD\uFF1A\u591A\u8BED\u79CD\u97F3\u9891\u53CD\u6B3A\u9A97\u6570\u636E\u96C6\u300B\uFF082024\uFF09\uFF0C\u4ECE\u8BBA\u6587\u9875\u9762\u83B7\u53D6\u6570\u636E\u5165\u53E3\uFF08\u901A\u5E38\u4E3A HuggingFace \u6570\u636E\u96C6\u9875\uFF09\u3002",
  "\u5728 HuggingFace \u7AD9\u5185\u641C\u7D22 MLAAD \u5173\u952E\u8BCD\u786E\u8BA4\u5B98\u65B9\u6570\u636E\u9875\u540E\u518D\u4E0B\u8F7D\uFF08\u5883\u5185\u53EF\u7528 HF_ENDPOINT=https://hf-mirror.com \u955C\u50CF\uFF09\u3002",
]));
bodyChildren.push(noteBox("\u590D\u6838\u6307\u5F15\uFF1A\u672C\u6B21\u68C0\u7D22\u672A\u80FD\u5B9A\u4F4D\u5230\u53EF\u76F4\u63A5\u9A8C\u8BC1\u7684\u5B98\u65B9\u4ED3\u5E93/\u6570\u636E\u9875\uFF0C\u4E0A\u8FF0\u4E24\u6B65\u4E3A\u6307\u5F15\u5F0F\u83B7\u53D6\uFF1B\u4E0B\u8F7D\u524D\u52A1\u5FC5\u5BF9\u7167\u8BBA\u6587\u4E0E\u9875\u9762\u539F\u6587\u786E\u8BA4\u771F\u4F2A\u6807\u7B7E\u7EA6\u5B9A\uFF0C\u907F\u514D\u8BEF\u7528\u540C\u540D\u975E\u5B98\u65B9\u8F6C\u8F7D\u3002", "warn"));

bodyChildren.push(h2("3.6 In-the-Wild\uFF08\u771F\u5B9E\u573A\u666F\u8DE8\u57DF\u8BC4\u4F30\uFF09"));
bodyChildren.push(body("\u7B80\u4ECB\uFF1A\u6765\u81EA\u4E92\u8054\u7F51\u516C\u5F00\u89C6\u9891\u7684\u771F\u5B9E\u8BF4\u8BDD\u4EBA\u771F/\u4F2A\u914D\u5BF9\uFF08\u9879\u76EE\u5185\u6587\u6863\u8BB0\u8F7D\u7EA6 2.0 \u4E07\u771F / 1.2 \u4E07\u4F2A\uFF0C\u4EE5\u8BBA\u6587\u4E3A\u51C6\uFF09\uFF0C\u7528\u4E8E\u68C0\u9A8C\u58F0\u5B66\u901A\u9053\u5728\u91CE\u5916\u57DF\u7684\u96F6\u6837\u672C\u6CDB\u5316\u80FD\u529B\uFF0C\u662F\u9636\u6BB5\u6027\u6C47\u62A5\u91CC\u8DE8\u57DF\u77E9\u9635\u5B9E\u9A8C\u7684\u5173\u952E\u8BC4\u4F30\u96C6\u3002"));
bodyChildren.push(...numberedList([
  "\u5B98\u65B9\u9875\u9762\u4E3A asvspoof.org \u7AD9\u5185 In-the-Wild \u4E13\u9875\uFF08\u5C5E ASVspoof \u5BB6\u65CF\u9879\u76EE\uFF0C\u83B7\u53D6\u6D41\u7A0B\u4E0E 3.1 \u76F8\u540C\uFF09\u3002",
  "\u5907\u9009\uFF1A\u68C0\u7D22\u8BBA\u6587\u300AAudio Anti-Spoofing Detection in the Wild\u300B\uFF08Odyssey 2024\uFF09\uFF0C\u6309\u8BBA\u6587\u7ED9\u51FA\u7684\u6570\u636E\u94FE\u63A5\u83B7\u53D6\u3002",
]));
bodyChildren.push(noteBox("\u590D\u6838\u6307\u5F15\uFF1A\u6240\u5C5E\u57DF\u540D asvspoof.org \u672C\u6B21\u4E0D\u53EF\u8FBE\uFF0C\u94FE\u63A5\u5F85\u590D\u6838\uFF1B\u5176\u4ED6\u6E20\u9053\u4E0B\u7684\u8F6C\u8F7D\u4E0D\u4F5C\u4E3A\u4F9D\u636E\u3002", "warn"));

bodyChildren.push(h2("3.7 PartialSpoof\uFF08\u90E8\u5206\u4F2A\u9020\uFF0C\u6D41\u5F0F\u673A\u5236\u5951\u5408\uFF09"));
bodyChildren.push(body("\u7B80\u4ECB\uFF1A\u5728\u771F\u5B9E\u8BED\u97F3\u4E2D\u4EC5\u66FF\u6362\u90E8\u5206\u8BCD\u7EA7\u7247\u6BB5\u7684\u90E8\u5206\u4F2A\u9020\u6570\u636E\u5E93\uFF08\u6E90\u81EA Interspeech 2021 \u5DE5\u4F5C\u7684\u540E\u7EED\u7CFB\u5217\uFF09\u3002\u672C\u9879\u76EE\u6D41\u5F0F\u5F15\u64CE\u4EE5 3 \u79D2\u7A97\u53E3\u6ED1\u52A8\u68C0\u6D4B\uFF0C\u4E0E\u90E8\u5206\u4F2A\u9020\u7684\u8BCD\u7EA7\u8FB9\u754C\u5929\u7136\u5951\u5408\uFF0C\u9002\u5408\u9A8C\u8BC1\u7A97\u53E3\u7EA7\u635F\u5931\u3002\u83B7\u53D6\u65B9\u5F0F\uFF1A\u68C0\u7D22\u8BBA\u6587\u300AAn Initial Investigation for Detecting Partially Spoofed Audio\u300B\u53CA\u5176\u540E\u7EED\u7248\u672C\uFF0C\u6309\u4F5C\u8005\u9875\u9762/\u4ED3\u5E93\u6307\u5F15\u7533\u8BF7\u4E0B\u8F7D\u3002"));
bodyChildren.push(noteBox("\u590D\u6838\u6307\u5F15\uFF1A\u672C\u6B21\u672A\u80FD\u5B9A\u4F4D\u5230\u53EF\u9A8C\u8BC1\u7684\u5B98\u65B9\u4ED3\u5E93\u5730\u5740\uFF0C\u4E0D\u63D0\u4F9B\u5177\u4F53\u94FE\u63A5\uFF1B\u8BF7\u4EE5\u8BBA\u6587\u5F15\u7528\u7684\u5B98\u65B9\u6E20\u9053\u4E3A\u51C6\u7533\u8BF7\u3002", "warn"));

// 第 4 章 声纹核验通道
bodyChildren.push(h1("\u7B2C 4 \u7AE0  \u58F0\u7EB9\u6838\u9A8C\u901A\u9053\u6570\u636E"));
bodyChildren.push(h2("4.1 AISHELL-3\uFF08\u4E2D\u6587\u591A\u8BF4\u8BDD\u4EBA\u8BED\u6599\uFF09"));
bodyChildren.push(body("\u7B80\u4ECB\uFF08\u5B98\u65B9\u9875\u539F\u6587\u5DF2\u6838\u5B9E\uFF09\uFF1A\u5927\u89C4\u6A21\u9AD8\u4FDD\u771F\u4E2D\u6587\u666E\u901A\u8BDD\u591A\u8BF4\u8BDD\u4EBA\u8BED\u6599\uFF0C\u7EA6 85 \u5C0F\u65F6\u60C5\u7EEA\u4E2D\u6027\u8BED\u97F3\u3001218 \u4F4D\u6BCD\u8BED\u8BF4\u8BDD\u4EBA\u300188,035 \u6761\u8BED\u53E5\uFF0C\u9644\u5B57\u7B26\u7EA7/\u62FC\u97F3\u7EA7\u8F6C\u5199\uFF08\u51C6\u786E\u7387 98% \u4EE5\u4E0A\uFF09\u4E0E\u6027\u522B/\u5E74\u9F84/\u53E3\u97F3\u6807\u6CE8\uFF0C\u8BB8\u53EF\u8BC1 Apache 2.0\u3002\u672C\u9879\u76EE\u7528\u4E8E CAMPPlus \u8BF4\u8BDD\u4EBA\u5D4C\u5165\u7684\u4E2D\u6587\u57DF\u5FAE\u8C03\u4E0E\u9608\u503C\u6807\u5B9A\uFF08\u5F53\u524D match 0.55 / reject 0.35 \u9608\u503C\u7684\u57DF\u5185\u6821\u51C6\u9A8C\u8BC1\uFF09\u3002"));
bodyChildren.push(...numberedList([
  "\u56FD\u5185\u955C\u50CF\u76F4\u63A5\u4E0B\u8F7D\uFF08\u7EA6 19 GB\uFF09\uFF1Awget https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz",
  "\u6216\u4ECE\u5B98\u65B9\u5217\u8868\u9875 openslr.org/93 \u9009\u62E9\u4EFB\u4E00\u955C\u50CF\uFF08EU \u8282\u70B9\u5728\u56FD\u5185\u8F83\u6162\uFF09\u3002",
  "\u6821\u9A8C\u6587\u4EF6\u6E05\u5355\u4E0E\u5B98\u65B9\u4E00\u81F4\u540E\u89E3\u538B\u81F3 D:/VeriCall_data/aishell3/\u3002",
]));
bodyChildren.push(h2("4.2 MUSAN\uFF08\u566A\u58F0/\u97F3\u4E50/\u8BED\u8A00\u589E\u5F3A\u8BED\u6599\uFF09"));
bodyChildren.push(body("\u7B80\u4ECB\uFF08\u5B98\u65B9\u9875\u539F\u6587\u5DF2\u6838\u5B9E\uFF09\uFF1A\u97F3\u4E50\u3001\u8BED\u8A00\u3001\u566A\u58F0\u4E09\u90E8\u5206\u589E\u5F3A\u8BED\u6599\u5408\u96C6\uFF0C\u5355\u5305 musan.tar.gz \u7EA6 11 GB\uFF0C\u8BB8\u53EF\u8BC1 CC BY 4.0\u3002\u672C\u9879\u76EE\u7528\u4E8E\u58F0\u7EB9\u6CE8\u518C\u8BED\u97F3\u7684\u52A0\u566A\u589E\u5F3A\u4E0E\u56F0\u96BE\u8D1F\u6837\u672C\u6784\u9020\u3002"));
bodyChildren.push(...numberedList([
  "\u56FD\u5185\u955C\u50CF\u4E0B\u8F7D\uFF1Awget https://openslr.magicdatatech.com/resources/17/musan.tar.gz",
  "\u4E5F\u53EF\u4ECE openslr.org/17 \u83B7\u53D6\u5168\u90E8\u8282\u70B9\u5217\u8868\u3002",
  "\u89E3\u538B\u81F3 D:/VeriCall_data/musan/\uFF0C\u4FDD\u7559 noise/music/speech \u4E09\u4E2A\u5B50\u76EE\u5F55\u7ED3\u6784\u3002",
]));
bodyChildren.push(h2("4.3 \u5BB6\u5EAD\u58F0\u7EB9\u81EA\u91C7\u534F\u8BAE"));
bodyChildren.push(body("\u5BB6\u5EAD\u6210\u5458\u58F0\u7EB9\u662F\u58F0\u7EB9\u901A\u9053\u7684\u6700\u7EC8\u670D\u52A1\u5BF9\u8C61\uFF0C\u5FC5\u987B\u6309\u7EDF\u4E00\u534F\u8BAE\u91C7\u96C6\uFF0C\u5E76\u4E8B\u5148\u7B7E\u7F72\u77E5\u60C5\u540C\u610F\u4E66\uFF08\u660E\u786E\u7528\u9014\u3001\u53EF\u968F\u65F6\u64A4\u56DE\u3001\u4EC5\u5B58\u50A8\u4E8E\u672C\u5730\uFF09\u3002"));
bodyChildren.push(...dataTable(
  ["\u9879\u76EE", "\u8981\u6C42"],
  [
    ["\u91C7\u96C6\u5BF9\u8C61", "\u76F4\u7CFB\u5BB6\u5EAD\u6210\u5458\uFF08\u7236\u6BCD\u3001\u5B50\u5973\u7B49\u9AD8\u9891\u8054\u7CFB\u4EBA\uFF09\uFF0C\u6BCF\u4EBA\u81F3\u5C11 3 \u4E2A\u53D7\u8BD5\u4EBA"],
    ["\u4F1A\u8BDD\u91CF", "\u6BCF\u4EBA\u2265 3 \u6B21\u4F1A\u8BDD\u00D7\u6BCF\u6B21\u2265 5 \u5206\u949F\uFF0C\u95F4\u9694\u4E0D\u540C\u65E5\u671F\u4E0E\u65F6\u6BB5"],
    ["\u4FE1\u9053\u8986\u76D6", "\u5B89\u9759\u5BA4\u5185\u4E0E\u7535\u8BDD\u514D\u63D0\u5404\u4E00\u7EC4\uFF08\u624B\u673A\u514D\u63D0\u901A\u8BDD\u5F55\u97F3\u6700\u63A5\u8FD1\u771F\u5B9E\u573A\u666F\uFF09"],
    ["\u5185\u5BB9\u8986\u76D6", "\u6570\u5B57\u4E32\u3001\u79F0\u8C13\u3001\u65E5\u5E38\u53E3\u8BED\u3001\u957F\u53E5\u5B50\uFF08\u4E0E ASR \u8F6C\u5199\u96BE\u5EA6\u68AF\u5EA6\u5BF9\u9F50\uFF09"],
    ["\u683C\u5F0F", "16 kHz\u3001\u5355\u58F0\u9053\u300116-bit PCM WAV\uFF1B\u547D\u540D speaker\u4F1A\u8BDD_\u65E5\u671F.wav"],
    ["\u5B58\u50A8", "D:/VeriCall_data/family_voiceprints/ \u52A0\u5BC6\u5B58\u50A8\uFF1B\u58F0\u7EB9\u5E93\u4EE5\u5D4C\u5165\u5411\u91CF\u5F62\u5F0F\u5165\u5E93\uFF0C\u539F\u59CB\u97F3\u9891\u4E0D\u8FDB\u5165\u4EE3\u7801\u4ED3\u5E93"],
  ],
  [22, 78],
  "\u8868 4-1  \u5BB6\u5EAD\u58F0\u7EB9\u91C7\u96C6\u534F\u8BAE"
));

// 第 5 章 语义通道
bodyChildren.push(h1("\u7B2C 5 \u7AE0  \u8BED\u4E49\u901A\u9053\u8BDD\u672F\u8BED\u6599"));
bodyChildren.push(h2("5.1 \u516C\u5F00\u8BDD\u672F\u6765\u6E90"));
bodyChildren.push(body("\u8BC8\u9A97\u8BDD\u672F\u65E0\u73B0\u6210\u516C\u5F00\u6570\u636E\u96C6\uFF0C\u9700\u4ECE\u5B98\u65B9\u901A\u62A5\u4E0E\u516C\u5F00\u5224\u4F8B\u4E2D\u7CFB\u7EDF\u6027\u6574\u7406\u3002\u5EFA\u8BAE\u6E20\u9053\uFF1A\u56FD\u5BB6\u53CD\u8BC8\u4E2D\u5FC3 App \u5178\u578B\u6848\u4F8B\u901A\u62A5\u3001\u516C\u5B89\u90E8\u5211\u4FA6\u5C40\u53CA\u5404\u5730\u53CD\u8BC8\u4E2D\u5FC3\u516C\u4F17\u53F7\u6848\u4F8B\u3001\u4E2D\u56FD\u88C1\u5224\u6587\u4E66\u7F51\u8BC8\u9A97\u7C7B\u5224\u51B3\u4E66\u4E2D\u7684\u8BDD\u672F\u7247\u6BB5\u3001\u4E3B\u6D41\u5A92\u4F53\u62AB\u9732\u7684\u5B8C\u6574\u5267\u672C\uFF08\u5192\u5145\u516C\u68C0\u6CD5\u3001\u5192\u5145\u719F\u4EBA\u3001\u5047\u5192\u5BA2\u670D\u9000\u6B3E\u7B49\u7C7B\u578B\uFF09\u3002\u6574\u7406\u4E3A JSONL\uFF0C\u6BCF\u884C\u5B57\u6BB5\uFF1A"));
bodyChildren.push(...codeBlock([
  '{"text": "\u4F60\u5148\u8F6C\u4E94\u4E07\u5230\u8FD9\u4E2A\u5B89\u5168\u8D26\u6237\uFF0C\u522B\u544A\u8BC9\u5BB6\u91CC\u4EBA",',
  ' "category": "\u5192\u5145\u516C\u68C0\u6CD5", "source": "\u53CD\u8BC8\u901A\u62A5", "date": "2026-08"}',
]));
bodyChildren.push(h2("5.2 \u672C\u5730 LLM \u6269\u589E\u6D41\u7A0B"));
bodyChildren.push(body("\u4E3A\u63D0\u5347\u8BDD\u672F\u8986\u76D6\u5EA6\uFF0C\u7528\u672C\u5730 Ollama \u7684 deepseek-r1:8b \u5BF9\u79CD\u5B50\u8BDD\u672F\u505A\u540C\u4E49\u6539\u5199\u4E0E\u89D2\u8272\u626E\u6F14\u53D8\u4F53\uFF08\u5982\u6539\u6362\u79F0\u8C13\u3001\u91D1\u989D\u3001\u7D27\u8FEB\u8BDD\u672F\u5F3A\u5EA6\uFF09\uFF0C\u6269\u589E\u540E\u7684\u8BED\u6599\u5FC5\u987B\u7ECF\u4EBA\u5DE5\u62BD\u68C0\u4E0D\u4F4E\u4E8E 20% \u540E\u5165\u5E93\uFF0C\u5E76\u4FDD\u7559\u6269\u589E\u6807\u8BB0\u4EE5\u4FBF\u6EAF\u6E90\u3002\u672C\u9879\u76EE\u89C4\u5219\u8BC4\u5206\u5668\uFF08rule_scorer\uFF09\u7684\u56DB\u7C7B\u89C4\u5219\u4E0E\u8BE5\u8BED\u6599\u5E93\u7684 category \u5B57\u6BB5\u5BF9\u9F50\u3002"));
bodyChildren.push(h2("5.3 \u8131\u654F\u7EA2\u7EBF"));
bodyChildren.push(bullet("\u5224\u51B3\u4E66/\u901A\u62A5\u4E2D\u7684\u771F\u5B9E\u59D3\u540D\u3001\u94F6\u884C\u5361\u53F7\u3001\u624B\u673A\u53F7\u7B49\u4E00\u5F8B\u66FF\u6362\u4E3A\u3010\u59D3\u540D\u3011\u3010\u5361\u53F7\u3011\u7B49\u5360\u4F4D\u7B26\u3002"));
bodyChildren.push(bullet("\u8BED\u6599\u4EC5\u7528\u4E8E\u9632\u5FA1\u6027\u7814\u7A76\u4E0E\u6A21\u578B\u8BC4\u4F30\uFF0C\u7981\u6B62\u5BF9\u5916\u5206\u4EAB\u542B\u672A\u8131\u654F\u4FE1\u606F\u7684\u539F\u6587\u3002"));
bodyChildren.push(bullet("\u6765\u6E90\u767B\u8BB0\u5230 source \u5B57\u6BB5\uFF0C\u4FDD\u7559\u53EF\u6EAF\u6E90\u4E0E\u53EF\u590D\u73B0\u6027\u3002"));

// 第 6 章 红队
bodyChildren.push(h1("\u7B2C 6 \u7AE0  \u7EA2\u961F\u6DF1\u5EA6\u4F2A\u9020\u8BED\u97F3\u5236\u4F5C"));
bodyChildren.push(h2("6.1 GPT-SoVITS\uFF08\u5C11\u6837\u672C\u514B\u9686\uFF0C\u4E2D\u6587\u6548\u679C\u4F73\uFF09"));
bodyChildren.push(body("\u5B98\u65B9 README \u5DF2\u6838\u5B9E\uFF1A\u5F3A\u5927\u7684\u5C11\u6837\u672C\u8BED\u97F3\u8F6C\u6362\u4E0E TTS WebUI\uFF0C5 \u79D2\u53C2\u8003\u97F3\u9891\u5373\u53EF\u96F6\u6837\u672C\u5408\u6210\uFF0C1 \u5206\u949F\u6570\u636E\u5FAE\u8C03\u53EF\u663E\u8457\u63D0\u5347\u76F8\u4F3C\u5EA6\uFF1B\u652F\u6301\u4E2D\u3001\u82F1\u3001\u65E5\u3001\u97E9\u3001\u7CA4\uFF1B\u8BB8\u53EF\u8BC1 MIT\uFF1B\u4ED3\u5E93 github.com/RVC-Boss/GPT-SoVITS\u3002"));
bodyChildren.push(...numberedList([
  "Windows \u63A8\u8350\u6574\u5408\u5305\uFF1A\u4ECE HuggingFace lj1995/GPT-SoVITS-windows-package \u4E0B\u8F7D 7z \u5305\uFF0C\u89E3\u538B\u540E\u8FD0\u884C go-webui.bat\uFF08\u56FD\u5185\u53EF\u8D70\u8BED\u96C0\u6587\u6863\u955C\u50CF\uFF09\u3002",
  "\u9884\u8BAD\u7EC3\u6A21\u578B\u4ECE HuggingFace lj1995/GPT-SoVITS \u4E0B\u8F7D\uFF0C\u653E\u5165 GPT_SoVITS/pretrained_models\u3002",
  "WebUI \u4E2D\u4E0A\u4F20\u5BB6\u5EAD\u6210\u5458\u53C2\u8003\u97F3\u9891\u4E0E\u6587\u672C\uFF08\u6587\u672C\u53D6\u81EA\u7B2C 5 \u7AE0\u8BDD\u672F\u8BED\u6599\uFF09\uFF0C\u63A8\u7406\u5BFC\u51FA 16 kHz WAV\u3002",
]));
bodyChildren.push(h2("6.2 CosyVoice\uFF08\u96F6\u6837\u672C\u8DE8\u8BED\u8A00\u514B\u9686\uFF09"));
bodyChildren.push(body("\u5B98\u65B9 README \u5DF2\u6838\u5B9E\uFF1A\u9762\u5411\u91CE\u5916\u573A\u666F\u7684\u96F6\u6837\u672C\u591A\u8BED\u8A00\u5408\u6210\u4E0E\u514B\u9686\uFF1B\u5B89\u88C5 git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\uFF1B\u6A21\u578B\u53CC\u6E20\u9053\uFF1A\u56FD\u5185 ModelScope\uFF08\u5982 iic/CosyVoice2-0.5B\uFF09\u3001\u6D77\u5916 HuggingFace\uFF08FunAudioLLM/*\uFF09\u3002\u6CE8\u610F\uFF1AREADME \u58F0\u660E\u5185\u5BB9\u4EC5\u4F9B\u5B66\u672F\u7528\u9014\uFF0C\u8BB8\u53EF\u8BC1\u4EE5\u4ED3\u5E93 LICENSE \u6587\u4EF6\u4E3A\u51C6\uFF0C\u4F7F\u7528\u524D\u81EA\u884C\u6838\u9605\u3002"));
bodyChildren.push(...numberedList([
  "conda \u521B\u5EFA Python 3.10 \u73AF\u5883\u540E\u6267\u884C\u4ED3\u5E93\u5185\u5B89\u88C5\u811A\u672C\uFF0C\u518D\u4E0B\u8F7D\u6A21\u578B\u81F3\u6307\u5B9A\u76EE\u5F55\u3002",
  "\u96F6\u6837\u672C\u6A21\u5F0F\u4E0B\u4F20\u5165\u53C2\u8003\u97F3\u9891 + \u76EE\u6807\u6587\u672C\uFF08\u8BDD\u672F\u8BED\u6599\uFF09\u751F\u6210\u4F2A\u9020\u6837\u672C\u3002",
]));
bodyChildren.push(h2("6.3 Seed-VC\uFF08\u96F6\u6837\u672C\u5B9E\u65F6\u8F6C\u6362\uFF09"));
bodyChildren.push(body("\u5B98\u65B9 README \u5DF2\u6838\u5B9E\uFF1A\u65E0\u9700\u8BAD\u7EC3\uFF0C\u7ED9\u5B9A 1~30 \u79D2\u53C2\u8003\u97F3\u9891\u5373\u53EF\u514B\u9686\u58F0\u97F3\uFF0C\u652F\u6301\u5B9E\u65F6\u8F6C\u6362\u4E0E\u6B4C\u58F0\u8F6C\u6362\uFF1B\u4ED3\u5E93 github.com/Plachtaa/seed-vc\uFF1B\u68C0\u67E5\u70B9\u9996\u6B21\u63A8\u7406\u65F6\u81EA\u52A8\u4ECE HuggingFace Plachta/Seed-VC \u4E0B\u8F7D\uFF0C\u7F51\u7EDC\u53D7\u9650\u65F6\u8BBE\u7F6E HF_ENDPOINT=https://hf-mirror.com\u3002README \u672A\u6807\u6CE8\u8BB8\u53EF\u8BC1\uFF0C\u4F7F\u7528\u524D\u81EA\u67E5 LICENSE\u3002\u9002\u5408\u6A21\u62DF\u7535\u8BDD\u4FE1\u9053\u5B9E\u65F6\u6CE8\u5165\u573A\u666F\u3002"));
bodyChildren.push(h2("6.4 \u7EA2\u961F\u5236\u4F5C SOP \u4E0E\u8D28\u91CF\u62BD\u68C0"));
bodyChildren.push(...numberedList([
  "\u514B\u9686\u76EE\u6807\u4EC5\u9650\u7B2C 4.3 \u8282\u81EA\u91C7\u5E76\u7B7E\u7F72\u540C\u610F\u4E66\u7684\u5BB6\u5EAD\u6210\u5458\u58F0\u97F3\uFF0C\u7981\u6B62\u4F7F\u7528\u4EFB\u4F55\u7B2C\u4E09\u65B9\u771F\u5B9E\u4EBA\u58F0\u3002",
  "\u5408\u6210\u6587\u672C\u53D6\u81EA\u7B2C 5 \u7AE0\u8BDD\u672F\u8BED\u6599\uFF0C\u6BCF\u4E2A TTS/VC \u5DE5\u5177\u751F\u6210\u4E0D\u4F4E\u4E8E 200 \u6761\uFF0C\u8986\u76D6\u4E0D\u540C\u65F6\u957F\u4E0E\u4FE1\u9053\u6761\u4EF6\u3002",
  "\u6309 \u6A21\u578B\u00D7\u65F6\u957F\u00D7\u4FE1\u9053 \u7F51\u683C\u62BD\u68C0\u4E3B\u89C2\u81EA\u7136\u5EA6\uFF08MOS \u2265 3\uFF09\uFF0C\u5254\u9664\u5408\u6210\u5931\u8D25\u6837\u672C\u3002",
  "\u4EA7\u7269\u5168\u90E8\u5B58\u653E\u4E8E D:/VeriCall_data/redteam/\uFF0C\u7981\u6B62\u8FDB\u5165\u4EFB\u4F55\u8BAD\u7EC3\u7BA1\u7EBF\uFF0C\u4EC5\u7528\u4E8E\u5168\u901A\u9053\u653B\u9632\u8BC4\u4F30\uFF08\u5BF9\u5E94 redteam_adversarial \u8BC4\u4F30\u6D41\u7A0B\uFF09\u3002",
]));

// 第 7 章 预处理
bodyChildren.push(h1("\u7B2C 7 \u7AE0  \u6570\u636E\u9884\u5904\u7406\u4E0E\u76EE\u5F55\u89C4\u8303"));
bodyChildren.push(h2("7.1 \u7EDF\u4E00\u8F6C\u7801"));
bodyChildren.push(body("\u6240\u6709\u5165\u5E93\u97F3\u9891\u7EDF\u4E00\u8F6C\u4E3A 16 kHz\u3001\u5355\u58F0\u9053\u300116-bit PCM WAV\uFF08\u4E0E AASIST \u7684 nb_samp=48000\uFF083 \u79D2@16 kHz\uFF09\u53CA CAMPPlus/SenseVoice \u91C7\u6837\u7387\u5BF9\u9F50\uFF09\uFF1A"));
bodyChildren.push(...codeBlock([
  "# \u6279\u91CF\u8F6C\u7801\uFF08Git Bash / Linux\uFF09",
  "find raw_dir -name \"*.wav\" | while read f; do",
  "  ffmpeg -y -i \"$f\" -ac 1 -ar 16000 -sample_fmt s16 \"out_dir/$(basename \"$f\")\"",
  "done",
]));
bodyChildren.push(h2("7.2 \u5207\u5206\u4E0E\u5143\u6570\u636E"));
bodyChildren.push(body("\u957F\u97F3\u9891\u6309\u9759\u97F3 VAD\uFF08ffmpeg silencedetect \u6216 webrtcvad\uFF09\u5207\u4E3A 3~6 \u79D2\u6BB5\uFF0C\u547D\u540D {spk}_{utt}_{seg:03d}.wav\uFF1B\u6BCF\u4E2A\u6570\u636E\u96C6\u7EF4\u62A4\u4E00\u4EFD meta.csv\uFF0C\u5B57\u6BB5\uFF1Apath, label(bonafide/spoof), attack_id, channel, source, license\u3002\u6807\u7B7E\u7EA6\u5B9A\u4E0E AASIST \u8BAD\u7EC3\u914D\u7F6E\u4E00\u81F4\uFF08bonafide=1 / spoof=0\uFF09\u3002"));
bodyChildren.push(h2("7.3 \u7EA2\u961F\u96C6\u9694\u79BB\u68C0\u67E5"));
bodyChildren.push(body("\u6BCF\u6B21\u8BC4\u4F30\u524D\u6267\u884C\u4E00\u6B21\u9694\u79BB\u68C0\u67E5\uFF1A\u786E\u8BA4 redteam/ \u4E0E in_the_wild/ \u3001partialspoof/ \u76EE\u5F55\u672A\u88AB\u4EFB\u4F55\u8BAD\u7EC3\u914D\u7F6E\u5F15\u7528\uFF0C\u907F\u514D\u8BC4\u4F30\u96C6\u6C61\u67D3\u8BAD\u7EC3\u96C6\u5BFC\u81F4\u6307\u6807\u865A\u9AD8\u3002"));

// 第 8 章 合规
bodyChildren.push(h1("\u7B2C 8 \u7AE0  \u5408\u89C4\u58F0\u660E\u4E0E\u81F4\u8C22"));
bodyChildren.push(h2("8.1 \u8BB8\u53EF\u8BC1\u6C47\u603B"));
bodyChildren.push(...dataTable(
  ["\u6570\u636E\u96C6", "\u8BB8\u53EF\u8BC1", "\u7528\u9014\u7EA6\u675F"],
  [
    ["AISHELL-3", "Apache 2.0\uFF08\u5DF2\u6838\u5B9E\uFF09", "\u5546\u7528\u9700\u9644\u8BB8\u53EF\u58F0\u660E"],
    ["MUSAN", "CC BY 4.0\uFF08\u5DF2\u6838\u5B9E\uFF09", "\u9700\u7F72\u540D\u5F15\u7528"],
    ["GPT-SoVITS", "MIT\uFF08\u5DF2\u6838\u5B9E\uFF09", "\u6309 MIT \u6761\u6B3E"],
    ["CosyVoice", "README \u58F0\u660E\u4EC5\u5B66\u672F\u7528\u9014\uFF0C\u4EE5 LICENSE \u6587\u4EF6\u4E3A\u51C6", "\u4EC5\u5B66\u672F\u7814\u7A76"],
    ["Seed-VC", "README \u672A\u6807\u6CE8\uFF0C\u4F7F\u7528\u524D\u81EA\u67E5 LICENSE", "\u5B66\u672F\u7814\u7A76"],
    ["ASVspoof / In-the-Wild", "EULA \u6CE8\u518C\u5236", "\u4EC5\u5B66\u672F\u7814\u7A76\uFF0C\u7981\u6B62\u518D\u5206\u53D1"],
    ["WaveFake / CFAD / FMFCC-A / MLAAD / PartialSpoof", "\u4EE5\u5404\u5B98\u65B9\u9875\u9762\u4E3A\u51C6", "\u5B66\u672F\u7814\u7A76"],
  ],
  [34, 36, 30],
  "\u8868 8-1  \u8BB8\u53EF\u8BC1\u4E0E\u7EA6\u675F\u6C47\u603B"
));
bodyChildren.push(h2("8.2 \u9879\u76EE\u58F0\u660E\u4E0E\u81F4\u8C22\u6A21\u677F"));
bodyChildren.push(body("\u9879\u76EE\u6210\u6790\u4E0E\u8BBA\u6587\u4E2D\u5EFA\u8BAE\u56FA\u5B9A\u4F7F\u7528\u5982\u4E0B\u58F0\u660E\uFF1A\u672C\u9879\u76EE\u7684\u8BC4\u4F30\u4F7F\u7528\u4E86 ASVspoof\u3001WaveFake\u3001FMFCC-A\u3001CFAD\u3001MUSAN\u3001AISHELL-3 \u7B49\u516C\u5F00\u6570\u636E\u96C6\uFF0C\u5747\u9075\u5FAA\u5404\u81EA\u8BB8\u53EF\u534F\u8BAE\u4EC5\u7528\u4E8E\u5B66\u672F\u7814\u7A76\uFF1B\u7EA2\u961F\u5BF9\u6297\u8BED\u97F3\u5168\u90E8\u57FA\u4E8E\u9879\u76EE\u56E2\u961F\u81EA\u91C7\u5E76\u83B7\u5F97\u6388\u6743\u7684\u5BB6\u5EAD\u6210\u5458\u58F0\u97F3\u5408\u6210\uFF0C\u672A\u4F7F\u7528\u4EFB\u4F55\u7B2C\u4E09\u65B9\u771F\u5B9E\u4EBA\u58F0\uFF1B\u8BC8\u9A97\u8BDD\u672F\u8BED\u6599\u5747\u7ECF\u8131\u654F\u5904\u7406\u3002\u8C28\u5411\u5404\u6570\u636E\u96C6\u7EC4\u7EC7\u4E0E\u4F5C\u8005\u81F4\u8C22\u3002"));

// ── 文档组装 ──
const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1417, bottom: 1417, left: 1701, right: 1417 };
function pageFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: PAL.secondary, font: FONT_BODY })],
    })],
  });
}

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT_BODY, size: 24, color: PAL.bodyColor },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: FONT_HEAD, size: 32, bold: true, color: PAL.headingColor },
        paragraph: { spacing: { before: 360, after: 160, line: 312 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: FONT_HEAD, size: 28, bold: true, color: PAL.headingColor },
        paragraph: { spacing: { before: 240, after: 120, line: 312 }, outlineLevel: 1 },
      },
      heading3: {
        run: { font: FONT_HEAD, size: 24, bold: true, color: PAL.headingColor },
        paragraph: { spacing: { before: 200, after: 100, line: 312 }, outlineLevel: 2 },
      },
    },
  },
  numbering: { config: numberingConfigs },
  sections: [
    { // 第 1 节：封面（无页码、无页脚）
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR1({
        title: "\u8C1B\u542C VeriCall \u6570\u636E\u83B7\u53D6\u65B9\u5F0F\u8BE6\u518C",
        subtitle: "\u4E09\u901A\u9053\u6570\u636E\u96C6\u6E05\u5355 \u00B7 \u83B7\u53D6\u6B65\u9AA4 \u00B7 \u5408\u89C4\u7EA2\u7EBF \u00B7 \u9884\u5904\u7406\u89C4\u8303",
        englishLabel: "DATA ACQUISITION GUIDE",
        metaLines: [
          "\u9879\u76EE\uFF1A\u8C1B\u542C VeriCall \u00B7 AI \u8BED\u97F3\u8BC8\u9A97\u62E6\u622A\u7CFB\u7EDF",
          "\u901A\u9053\uFF1A\u58F0\u5B66\u9632\u4F2A \u00B7 \u58F0\u7EB9\u6838\u9A8C \u00B7 \u8BED\u4E49\u5206\u6790 \u00B7 \u7EA2\u961F\u5BF9\u6297",
          "\u94FE\u63A5\u6838\u5B9E\uFF1A2026-09-03\uFF088 \u9879\u5DF2\u6838\u5B9E \u00B7 4 \u9879\u5F85\u590D\u6838\uFF09",
        ],
        footerLeft: "\u8C1B\u542C VeriCall \u9879\u76EE\u7EC4",
        footerRight: "2026 \u5E74 9 \u6708",
        palette: { bg: PAL.bg, accent: PAL.accent, ...PAL.cover },
      }),
    },
    { // 第 2 节：目录（罗马页码）
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      footers: { default: pageFooter() },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { before: 480, after: 360 },
          children: [new TextRun({ text: "\u76EE  \u5F55", bold: true, size: 32, font: FONT_HEAD, color: PAL.headingColor })],
        }),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
        new Paragraph({
          spacing: { before: 200 },
          children: [new TextRun({
            text: "\u6CE8\uFF1A\u672C\u76EE\u5F55\u7531\u57DF\u4EE3\u7801\u751F\u6210\u3002\u5982\u7F16\u8F91\u540E\u9875\u7801\u53D8\u52A8\uFF0C\u8BF7\u5728\u76EE\u5F55\u4E0A\u53F3\u952E\u9009\u62E9\u201C\u66F4\u65B0\u57DF\u201D\u4EE5\u5237\u65B0\u9875\u7801\u3002",
            italics: true, size: 18, color: "888888", font: FONT_BODY,
          })],
        }),
      ],
    },
    { // 第 3 节：正文（阿拉伯页码从 1 起）
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      footers: { default: pageFooter() },
      children: bodyChildren,
    },
  ],
});

const OUT = process.argv[2] || "output.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE " + OUT + " (" + buf.length + " bytes)");
}).catch(e => { console.error(e); process.exit(1); });
