"""
pipeline/router.py

Шаг 4 — Маршрутизация вопросов + сборка контекста.

Каждый вопрос из промпта привязан к типам документов и ключевым словам.
Разделы ранжируются по релевантности, затем набираются до лимита токенов.
"""

from dataclasses import dataclass
from pipeline.classifier import TYPE_PRIORITY
from pipeline.splitter import Section


# Ключевые слова по теме вопросов (стемы, без окончаний)
SECTION_KEYWORDS: list[str] = [
    # Срок поставки
    "срок", "поставк", "исполнен", "дней", "календарн", "рабочих дн",
    # Место поставки
    "место поставк", "адрес", "получател", "склад", "доставк",
    # Обеспечение
    "обеспечен", "банковск", "гарантия", "залог",
    # Монтаж
    "монтаж", "установк", "пусконаладк", "запуск", "сборк", "шефмонтаж",
    # Документы при поставке
    "сертификат", "паспорт", "удостоверен", "свидетельств", "поверк",
    "регистрационн", "накладн", "счёт-фактур", "товарн",
    # Штрафы
    "штраф", "пен ", "неустойк", "санкц", "ответственност",
    # Оплата
    "оплат", "расчёт", "расчет", "платёж", "платеж", "финансиров",
    # Национальный режим
    "нацрежим", "национальн", "1875", "925",
    "запрет", "ограничен", "преимуществ",
    "постановлен", "правительств",
    "страна происхожден", "подтвержден", "производств",
]


@dataclass
class RankedSection:
    doc_name: str
    doc_type: str
    section: Section
    score: float


def _score_section(section: Section, doc_type: str) -> float:
    """
    Оценивает релевантность раздела.
    Учитывает: ключевые слова + приоритет типа документа.
    """
    text_lower = section.text.lower()
    keyword_hits = sum(1 for kw in SECTION_KEYWORDS if kw in text_lower)

    # Штраф за мусорный тип
    type_penalty = TYPE_PRIORITY.get(doc_type, 5)

    # Базовый score: ключевые слова минус штраф за неважный тип
    return keyword_hits - (type_penalty * 0.3)


def build_context(
    docs: list[dict],  # [{"name": str, "type": str, "sections": list[Section]}]
    max_chars: int = 100_000,
) -> tuple[str, bool, int]:
    """
    Собирает финальный контекст для промпта.

    Returns:
        context (str): текст для подстановки в промпт
        truncated (bool): был ли контекст урезан
        char_count (int): итоговое количество символов
    """
    # Исключаем мусор
    useful_docs = [d for d in docs if d["type"] != "МУСОР"]
    if not useful_docs:
        useful_docs = docs  # если всё мусор — берём всё

    # Ранжируем все разделы
    ranked: list[RankedSection] = []
    for doc in useful_docs:
        for section in doc["sections"]:
            score = _score_section(section, doc["type"])
            ranked.append(RankedSection(
                doc_name=doc["name"],
                doc_type=doc["type"],
                section=section,
                score=score,
            ))

    # Сначала по убыванию score, потом документы с более высоким приоритетом типа
    ranked.sort(key=lambda r: (-r.score, TYPE_PRIORITY.get(r.doc_type, 5)))

    context_parts: list[str] = []
    total_chars = 0
    truncated = False

    for ranked_sec in ranked:
        block = (
            f"\n\n--- [{ranked_sec.doc_name} | {ranked_sec.doc_type}] "
            f"Раздел: {ranked_sec.section.heading} ---\n"
            f"{ranked_sec.section.text}"
        )

        if total_chars + len(block) > max_chars:
            # Если раздел очень релевантный — вставляем его обрезанным
            if ranked_sec.score >= 3:
                remaining = max_chars - total_chars - 120
                if remaining > 300:
                    block = block[:remaining] + "\n...[обрезано]"
                    context_parts.append(block)
                    total_chars += len(block)
            truncated = True
            break

        context_parts.append(block)
        total_chars += len(block)

    context = "".join(context_parts).strip()
    return context, truncated, total_chars


# -- Принудительный поиск нацрежима ----------------------------------------

NACREG_KEYWORDS = ["1875", "925", "нацрежим", "национальн", "запрет", "ограничен", "преимуществ"]


def extract_nacreg_paragraphs(docs: list[dict]) -> str:
    """
    Ищет абзацы про нацрежим по ВСЕМ документам.
    Возвращает найденные абзацы как текст для вставки в контекст.
    """
    found: list[str] = []
    seen: set[str] = set()

    for doc in docs:
        for section in doc["sections"]:
            # Разбиваем раздел на абзацы
            paragraphs = section.text.split("\n")
            for para in paragraphs:
                para_lower = para.lower().strip()
                if len(para_lower) < 20:
                    continue
                # Ищем ключевые слова нацрежима
                hits = sum(1 for kw in NACREG_KEYWORDS if kw in para_lower)
                if hits >= 1:
                    # Дедупликация
                    key = para_lower[:80]
                    if key not in seen:
                        seen.add(key)
                        found.append(para.strip())

    if not found:
        return ""

    return "\n\n--- [НАЦРЕЖИМ — принудительно извлечённые абзацы] ---\n" + "\n".join(found)


# -- Типовые ключевые слова для контекстов по типам документов ----------------

CONTRACT_KEYWORDS: list[str] = [
    "штраф", "пен", "неустойк", "санкц", "ответственност",
    "оплат", "расчёт", "расчет", "платёж", "платеж",
    "обеспечен", "банковск", "гарантия", "залог",
    "исполнен", "приёмк", "приемк", "срок", "поставк",
    "документ", "накладн", "сертификат", "паспорт",
    "качеств", "верхн", "предел", "не более",
    "актуальн", "интерес", "отказ",
]

NOTICE_KEYWORDS: list[str] = [
    "нацрежим", "национальн", "1875", "925",
    "запрет", "ограничен", "преимуществ",
    "обеспечен", "гарантия", "банковск",
    "казначейск", "гособоронзаказ", "гоз", "275-фз", "упз",
    "аналог", "эквивалент",
    "начальн", "максимальн", "нмц",
]

TECHSPEC_KEYWORDS: list[str] = [
    "место поставк", "адрес", "получател", "склад", "доставк",
    "гарантий", "гарантия", "срок",
    "документ", "сертификат", "паспорт", "поверк", "удостоверен",
    "аналог", "эквивалент",
    "казначейск", "гособоронзаказ", "гоз", "275-фз", "упз",
]


def build_context_for_type(
    docs: list[dict],
    target_types: list[str],
    max_chars: int = 60_000,
    keywords: list[str] | None = None,
) -> tuple[str, bool, int]:
    """
    Строит контекст только из документов указанных типов.

    Args:
        docs: все документы [{name, type, sections, char_count}]
        target_types: ["КОНТРАКТ"], ["ИЗВЕЩЕНИЕ", "ТРЕБОВАНИЯ"], ["ТЗ"]
        max_chars: лимит символов
        keywords: ключевые слова для ранжирования (если None — SECTION_KEYWORDS)

    Returns:
        (context, truncated, char_count)
    """
    filtered = [d for d in docs if d["type"] in target_types]
    if not filtered:
        return "", False, 0

    kw = keywords or SECTION_KEYWORDS

    ranked: list[RankedSection] = []
    for doc in filtered:
        for section in doc["sections"]:
            text_lower = section.text.lower()
            score = sum(1 for k in kw if k in text_lower)
            ranked.append(RankedSection(
                doc_name=doc["name"],
                doc_type=doc["type"],
                section=section,
                score=score,
            ))

    ranked.sort(key=lambda r: -r.score)

    context_parts: list[str] = []
    total_chars = 0
    truncated = False

    for ranked_sec in ranked:
        block = (
            f"\n\n--- [{ranked_sec.doc_name}] "
            f"Раздел: {ranked_sec.section.heading} ---\n"
            f"{ranked_sec.section.text}"
        )

        if total_chars + len(block) > max_chars:
            if ranked_sec.score >= 2:
                remaining = max_chars - total_chars - 120
                if remaining > 300:
                    block = block[:remaining] + "\n...[обрезано]"
                    context_parts.append(block)
                    total_chars += len(block)
            truncated = True
            break

        context_parts.append(block)
        total_chars += len(block)

    context = "".join(context_parts).strip()
    return context, truncated, total_chars
