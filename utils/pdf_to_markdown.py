import pdfplumber
from tabulate import tabulate
import re


def clean_text(text):
    """清理文本：去除多余空白，保留段落结构"""
    if not text:
        return ""
    # 合并多个空行为单个空行，保留段落间距
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # 去除每行首尾空格
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)


def table_to_markdown(table):
    """将 pdfplumber 提取的表格（list of lists）转为 Markdown 表格"""
    if not table or not any(row for row in table):
        return ""

    # 过滤掉全空行
    cleaned_table = [row for row in table if any(cell and str(cell).strip() for cell in row)]
    if not cleaned_table:
        return ""

    # 使用 tabulate 生成 GitHub 风格 Markdown 表格
    return tabulate(cleaned_table, headers="firstrow", tablefmt="github")


def convert_pdf_to_markdown(pdf_path, md_path):
    """将 PDF 转为 Markdown 文件"""
    full_md = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            full_md += f"<!-- Page {page_num} -->\n\n"

            # 1. 提取所有表格及其在页面中的位置（用于避免重复提取）
            tables = page.extract_tables()
            table_boxes = []
            if hasattr(page, 'tables'):
                # 新版 pdfplumber 可获取表格 bbox
                for table in page.find_tables():
                    table_boxes.append(table.bbox)
            else:
                # 回退：用 extract_tables 的结果估算（不精确）
                pass

            # 2. 提取非表格文本（可选：排除表格区域）
            # 简化处理：先提取全文本，再手动插入表格（可能有重复）
            # 更优方案：使用 page.without_images().extract_text(...) 并跳过表格区域（复杂）
            text = page.extract_text()
            if text:
                full_md += clean_text(text) + "\n\n"

            # 3. 插入表格（放在页面末尾或根据位置插入）
            for table in tables:
                md_table = table_to_markdown(table)
                if md_table:
                    full_md += md_table + "\n\n"

            full_md += "\n---\n\n"  # 页面分隔符（可选）

    # 写入 Markdown 文件
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_md)


