"""
pipeline/splitter.py

Шаг 3 — Структурная нарезка.

Docling отдаёт Markdown с заголовками (## Раздел 1, ### п. 1.1).
Разбиваем по заголовкам → карта {заголовок → текст раздела}.
"""

import re
from dataclasses import dataclass


@dataclass
class Section:
    heading: str   # Текст заголовка (или "НАЧАЛО" для текста до первого заголовка)
    text: str      # Полный текст раздела включая заголовок
    level: int     # Уровень заголовка (1=# 2=## 3=### 0=нет)


def split_into_sections(markdown_text: str, min_section_chars: int = 80) -> list[Section]:
    """
    Разбивает Markdown на разделы по заголовкам.
    Возвращает список Section, отфильтровав слишком короткие.
    """
    # Паттерн: строки начинающиеся с # (Markdown заголовки)
    heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

    sections: list[Section] = []
    matches = list(heading_pattern.finditer(markdown_text))

    if not matches:
        # Нет Markdown-заголовков — пробуем нарезать по нумерованным заголовкам
        return _split_by_numbered_headings(markdown_text, min_section_chars)

    # Текст до первого заголовка
    preamble = markdown_text[:matches[0].start()].strip()
    if len(preamble) >= min_section_chars:
        sections.append(Section(heading="НАЧАЛО ДОКУМЕНТА", text=preamble, level=0))

    # Разделы по заголовкам
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        level = len(match.group(1))
        heading = match.group(2).strip()
        text = markdown_text[start:end].strip()

        if len(text) >= min_section_chars:
            sections.append(Section(heading=heading, text=text, level=level))

    return sections


def _split_by_numbered_headings(text: str, min_chars: int) -> list[Section]:
    """
    Резервный сплиттер для текста без Markdown-заголовков.
    Ищет нумерованные разделы: "1.", "1.1", "Раздел 1", "СТАТЬЯ 1", "Глава 1"
    """
    pattern = re.compile(
        r'(?:^|\n)(?=[ \t]*(?:\d+[\.\)]\s{1,4}[А-ЯA-Zа-яa-z]'
        r'|\bРаздел\b\s*\d'
        r'|\bГлава\b\s*\d'
        r'|\bСТАТЬЯ\b'
        r'|\bARTICLE\b))',
        re.IGNORECASE
    )

    parts = pattern.split(text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) >= min_chars]

    if len(parts) <= 1:
        # Совсем нет структуры — возвращаем как один блок
        return [Section(heading="ДОКУМЕНТ", text=text.strip(), level=0)] if text.strip() else []

    return [Section(heading=p.splitlines()[0][:80], text=p, level=1) for p in parts]
