"""
pipeline/extractor.py

Шаг 1 - Распаковка и инвентаризация
Шаг 2 - Извлечение текста:
         PDF     - pymupdf4llm (Markdown)
         DOCX    - python-docx (параграфы + таблицы)
         HTML    - BeautifulSoup
         TXT     - open().read()
         XLSX    - openpyxl
         XLS     - xlrd
         DOC     - antiword (системная утилита)

Поддерживает: PDF, DOCX, DOC, HTML, TXT, XLSX, XLS, ZIP (включая вложенные).
"""

import zipfile
import tempfile
import shutil
import subprocess
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SKIP_EXTENSIONS = {".pptx", ".xlsb", ".jpg", ".jpeg", ".png", ".gif", ".bmp"}

MAX_ZIP_DEPTH = 3


@dataclass
class ExtractedFile:
    name: str
    path: str
    text: str
    doc_type: str = "НЕИЗВЕСТНО"
    skipped: bool = False
    skip_reason: str = ""


# -- ZIP ------------------------------------------------------------------

def _fix_zip_filename(name: str) -> str:
    """Исправляет кодировку имён файлов в ZIP (CP437 -> CP866 для кириллицы)."""
    try:
        # Если имя содержит нечитаемые символы — скорее всего CP437, пробуем CP866
        if any(ord(c) > 127 for c in name):
            fixed = name.encode("cp437").decode("cp866")
            if any("\u0400" <= c <= "\u04FF" for c in fixed):  # Есть кириллица
                return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return name


def _unpack_zip(zip_path: Path, target_dir: Path, depth: int = 0) -> list[Path]:
    """Рекурсивно распаковывает ZIP, возвращает список путей к файлам."""
    if depth > MAX_ZIP_DEPTH:
        logger.warning(f"[ZIP] Превышена глубина вложенности: {zip_path}")
        return []

    found = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                # Исправляем кодировку имени
                fixed_name = _fix_zip_filename(info.filename)
                # Извлекаем файл
                data = zf.read(info.filename)
                dest = target_dir / fixed_name
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
        logger.info(f"[ZIP] Распакован: {zip_path.name}")
    except zipfile.BadZipFile:
        logger.warning(f"[ZIP] Битый архив: {zip_path}")
        return []

    for item in target_dir.rglob("*"):
        if item.is_file():
            if item.suffix.lower() == ".zip":
                nested_dir = item.parent / (item.stem + "_extracted")
                nested_dir.mkdir(exist_ok=True)
                found.extend(_unpack_zip(item, nested_dir, depth + 1))
            else:
                found.append(item)
    return found


# -- Обработчики по форматам ----------------------------------------------

def _extract_pdf(file_path: Path) -> str:
    """PDF -> Markdown через pymupdf4llm."""
    import pymupdf4llm
    logger.info(f"[PDF] pymupdf4llm -> {file_path.name}")
    text = pymupdf4llm.to_markdown(str(file_path))
    logger.info(f"[PDF] {len(text)} симв. из {file_path.name}")
    return text


def _extract_docx(file_path: Path) -> str:
    """DOCX -> текст через python-docx (параграфы + таблицы)."""
    from docx import Document
    logger.info(f"[DOCX] python-docx -> {file_path.name}")
    doc = Document(str(file_path))
    parts: list[str] = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    result = "\n".join(parts)
    logger.info(f"[DOCX] {len(result)} симв. из {file_path.name}")
    return result


def _extract_html(file_path: Path) -> str:
    """HTML -> текст через BeautifulSoup."""
    from bs4 import BeautifulSoup
    logger.info(f"[HTML] BeautifulSoup -> {file_path.name}")
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    logger.info(f"[HTML] {len(text)} симв. из {file_path.name}")
    return text


def _extract_txt(file_path: Path) -> str:
    """TXT -> просто читаем файл."""
    logger.info(f"[TXT] open() -> {file_path.name}")
    text = file_path.read_text(encoding="utf-8", errors="replace")
    logger.info(f"[TXT] {len(text)} симв. из {file_path.name}")
    return text


def _extract_xlsx(file_path: Path) -> str:
    """XLSX -> текст через openpyxl."""
    import openpyxl
    logger.info(f"[XLSX] openpyxl -> {file_path.name}")
    wb = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
    parts: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"--- Лист: {sheet_name} ---")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            line = " | ".join(cells).strip()
            if line.replace("|", "").strip():
                parts.append(line)
    wb.close()
    result = "\n".join(parts)
    logger.info(f"[XLSX] {len(result)} симв. из {file_path.name}")
    return result


def _extract_xls(file_path: Path) -> str:
    """XLS (старый формат) -> текст через xlrd."""
    import xlrd
    logger.info(f"[XLS] xlrd -> {file_path.name}")
    wb = xlrd.open_workbook(str(file_path))
    parts: list[str] = []
    for sheet in wb.sheets():
        parts.append(f"--- Лист: {sheet.name} ---")
        for row_idx in range(sheet.nrows):
            cells = [str(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)]
            line = " | ".join(cells).strip()
            if line.replace("|", "").strip():
                parts.append(line)
    result = "\n".join(parts)
    logger.info(f"[XLS] {len(result)} симв. из {file_path.name}")
    return result


def _extract_doc_antiword(file_path: Path) -> str:
    """DOC (OLE2) -> текст через antiword. Фолбэк: если .doc — на самом деле HTML."""
    logger.info(f"[DOC] antiword -> {file_path.name}")
    try:
        result = subprocess.run(
            ["antiword", str(file_path)],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "antiword вернул ненулевой код")
        text = result.stdout.strip()
        if not text:
            raise RuntimeError("antiword вернул пустой текст")
        logger.info(f"[DOC] {len(text)} симв. из {file_path.name}")
        return text
    except (RuntimeError, FileNotFoundError) as e:
        logger.warning(f"[DOC] antiword не смог: {e}. Пробую как HTML...")
        # Фолбэк: ЕИС часто отдаёт HTML-файлы с расширением .doc
        return _extract_html(file_path)


# Маппинг расширения -> обработчик
FORMAT_HANDLERS = {
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
    ".html": _extract_html,
    ".htm":  _extract_html,
    ".txt":  _extract_txt,
    ".xlsx": _extract_xlsx,
    ".xls":  _extract_xls,
    ".doc":  _extract_doc_antiword,
}


# -- Главная функция ------------------------------------------------------

def extract_files(uploaded_paths: list[Path]) -> list[ExtractedFile]:
    """
    Принимает список путей к загруженным файлам (могут быть ZIP).
    Возвращает список ExtractedFile с текстом.
    """
    results: list[ExtractedFile] = []
    all_files: list[Path] = []
    tmp_dirs: list[Path] = []

    # --- Шаг 1: инвентаризация ---
    for path in uploaded_paths:
        ext = path.suffix.lower()
        if ext == ".zip":
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_dirs.append(tmp_dir)
            unpacked = _unpack_zip(path, tmp_dir)
            all_files.extend(unpacked)
            logger.info(f"[ИНВЕНТАРИЗАЦИЯ] ZIP {path.name}: {len(unpacked)} файлов")
        else:
            all_files.append(path)

    logger.info(f"[ИНВЕНТАРИЗАЦИЯ] Всего файлов: {len(all_files)}")

    # --- Шаг 2: извлечение текста ---
    for file_path in all_files:
        ext = file_path.suffix.lower()
        name = file_path.name

        # Пропускаем служебные файлы из ZIP
        if any(part.startswith("__") or part.startswith(".") for part in file_path.parts):
            logger.info(f"[ПРОПУСК] Служебный: {name}")
            continue

        if ext in SKIP_EXTENSIONS:
            logger.info(f"[ПРОПУСК] {ext}: {name}")
            continue

        handler = FORMAT_HANDLERS.get(ext)
        if not handler:
            logger.info(f"[ПРОПУСК] Неизвестный формат {ext}: {name}")
            continue

        try:
            text = handler(file_path)
            if len(text.strip()) < 50:
                logger.warning(f"[ВНИМАНИЕ] {name}: текст короткий ({len(text)} симв.)")
            results.append(ExtractedFile(name=name, path=str(file_path), text=text))
        except Exception as e:
            logger.error(f"[ОШИБКА] {name}: {e}")
            logger.error(traceback.format_exc())
            results.append(ExtractedFile(
                name=name, path=str(file_path), text="",
                skipped=True, skip_reason=f"Ошибка: {str(e)[:120]}"
            ))

    # Очистка временных директорий
    for tmp_dir in tmp_dirs:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    ok = sum(1 for r in results if not r.skipped)
    skip = sum(1 for r in results if r.skipped)
    logger.info(f"[ИТОГ] Обработано: {len(results)}, успешно: {ok}, ошибки: {skip}")
    return results
