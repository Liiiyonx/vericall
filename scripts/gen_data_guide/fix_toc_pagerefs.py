# -*- coding: utf-8 -*-
"""fix_toc_pagerefs.py — 重算《数据获取方式详册》目录缓存页码并回写 docx。

背景：add_toc_placeholders 注入的 PAGEREF 缓存值来自旧版本排版，
正文改动后页码漂移（如 8.2 缓存 16，实际 12）。
本脚本：
  1. 用 LibreOffice 将 docx 转 PDF（临时目录，ASCII 路径）；
  2. 从 PDF 文本层定位每个目录标题所在页（正文页码 = PDF 页 - 前置页数）；
  3. 按目录条目顺序回写 PAGEREF separate/end 之间的缓存数字；
  4. 保持 docx 其余部分字节不变。

用法：
  python fix_toc_pagerefs.py <docx路径> [--front-pages 3]

前置页数 = 封面(1) + 目录页数(2) = 3（正文第 1 页 = PDF 第 4 页）。
"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"

# 目录 34 条目的标题定位关键字（与 generate.js 章节标题一致，顺序即目录顺序）
TOC_KEYS = [
    "第 1 章", "1.1 编制目的", "1.2 数据集核实",
    "第 2 章", "2.1 三通道", "2.2 存储规划",
    "第 3 章", "3.1 ASVspoof", "3.2 WaveFake", "3.3 FMFCC-A",
    "3.4 CFAD", "3.5 MLAAD", "3.6 In-the-Wild", "3.7 PartialSpoof",
    "第 4 章", "4.1 AISHELL-3", "4.2 MUSAN", "4.3 家庭声纹",
    "第 5 章", "5.1 公开话术", "5.2 本地", "5.3 脱敏",
    "第 6 章", "6.1 GPT-SoVITS", "6.2 CosyVoice", "6.3 Seed-VC", "6.4 红队制作",
    "第 7 章", "7.1 统一转码", "7.2 切分", "7.3 红队集隔离",
    "第 8 章", "8.1 许可证", "8.2 项目声明",
]

# 章节标题的精确正则（避免 "按第 7 章规范" 这类交叉引用误匹配）
def heading_pattern(key: str) -> re.Pattern:
    k = re.escape(re.sub(r"\s+", " ", key))
    if k.startswith("第"):
        # 章标题：行首“第 N 章”后跟两个空格与章名（排除“按第 7 章规范”式交叉引用）
        return re.compile(r"(?:^|\n)" + k + r"\s{2,}\S")
    # 节标题：行首起前缀匹配即可（键中含空格与标题文字，无歧义）
    return re.compile(r"(?:^|\n)" + k)


def main() -> int:
    docx_path = Path(sys.argv[1])
    front_pages = 3
    if "--front-pages" in sys.argv:
        front_pages = int(sys.argv[sys.argv.index("--front-pages") + 1])

    tmp = Path(tempfile.mkdtemp(prefix="tocfix_"))
    try:
        local = tmp / "guide.docx"
        shutil.copy(docx_path, local)
        subprocess.run(
            [SOFFICE, "--headless", "--convert-to", "pdf", str(local), "--outdir", str(tmp)],
            check=True, capture_output=True,
        )
        pdf = tmp / "guide.pdf"
        if not pdf.exists():
            print("ERROR: PDF 转换失败")
            return 1

        import pymupdf  # docxenv 环境提供
        doc = pymupdf.open(str(pdf))
        page_texts = []
        for pno in range(len(doc)):
            t = doc[pno].get_text().replace("　", " ")
            page_texts.append(t)

        # 逐条定位：从上一条的页开始向后找，保证顺序单调
        pages = []
        start = front_pages  # 0-based：正文起始 PDF 页索引
        for key in TOC_KEYS:
            pat = heading_pattern(key)
            hit = None
            for pno in range(start, len(doc)):
                if pat.search(page_texts[pno]):
                    hit = pno
                    break
            if hit is None:
                print(f"ERROR: 未定位到标题 {key!r}")
                return 1
            pages.append(hit + 1 - front_pages)  # 正文页码（1 起）
            start = hit  # 后续标题不早于本页

        print("目录页码映射:", pages)

        # 回写 docx 中 PAGEREF 缓存值
        with zipfile.ZipFile(docx_path, "r") as z:
            names = z.namelist()
            data = {n: z.read(n) for n in names}
        xml = data["word/document.xml"].decode("utf-8")

        paras = re.findall(r"<w:p\b.*?</w:p>", xml, re.S)
        toc_paras = [p for p in paras if "PAGEREF" in p]
        if len(toc_paras) != len(pages):
            print(f"ERROR: 目录条目数 {len(toc_paras)} != 映射数 {len(pages)}")
            return 1

        idx = 0
        def repl(m: re.Match) -> str:
            nonlocal idx
            para = m.group(0)
            if "PAGEREF" not in para:
                return para
            new_num = str(pages[idx])
            idx += 1
            return re.sub(
                r'(<w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>)\d+(</w:t>)',
                lambda mm: mm.group(1) + new_num + mm.group(2),
                para, count=1,
            )

        xml_new = re.sub(r"<w:p\b.*?</w:p>", repl, xml, flags=re.S)
        if idx != len(pages):
            print(f"ERROR: 仅替换 {idx} 条")
            return 1
        data["word/document.xml"] = xml_new.encode("utf-8")

        backup = docx_path.with_suffix(".docx.bak")
        shutil.copy(docx_path, backup)
        with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
            for n in names:
                z.writestr(n, data[n])
        print(f"OK: 已回写 {len(pages)} 条目录页码；备份于 {backup}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
