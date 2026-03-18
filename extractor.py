"""
pipeline/extractor.py

Шаг 1 — Распаковка и инвентаризация
Шаг 2 — Извлечение текста через Docling (PDF, DOCX, HTML, TXT)
         и antiword (старый .doc OLE2)

Поддерживает: PDF, DOCX, DOC, HTML, TXT, ZIP (включая вложенные ZIP).
"""

import zipfile
import tempfile
import shutil
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# Форматы, которые обрабатывает Docling
DOCLING_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt"}

# Форматы, которые читаем через antiword
ANTIWORD_EXTENSIONS = {".doc"}

# Всё остальное молча пропускаем
SKIP_EXTENSIONS = {".xls", ".xlsx", ".pptx", ".jpg", ".jpeg", ".png", ".gif", ".bmp"}

MAX_ZIP_DEPTH = 3  # Защита от zip-бомб


@dataclass
class ExtractedFile:
    name: str          # Имя файла (оригинальное)
    path: str          # Откуда взят (для отладки)
    text: str          # Извлечённый текст
    doc_type: str = "НЕИЗВЕСТНО"
    skipped: bool = False
    skip_reason: str = ""


def _unpack_zip(zip_path: Path, target_dir: Path, depth: int = 0) -> list[Path]:
    """Рекурсивно распаковывает ZIP, возвращает список путей к файлам."""
    if depth > MAX_ZIP_DEPTH:
        logger.warning(f"Превышена глубина вложенности ZIP: {zip_path}")
        return []

    found = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile:
        logger.warning(f"Не удалось распаковать {zip_path} — не является ZIP")
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


def _extract_doc_antiword(file_path: Path) -> str:
    """
    Извлекает текст из старого .doc через antiword.
    antiword устанавливается в Dockerfile как системный пакет.
    """
    try:
        result = subprocess.run(
            ["antiword", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(stderr or "antiword вернул ненулевой код")
        text = result.stdout.strip()
        if not text:
            raise RuntimeError("antiword вернул пустой текст — файл защищён паролем или повреждён")
        return text
    except FileNotFoundError:
        raise RuntimeError("antiword не найден. Убедитесь, что он установлен (см. Dockerfile)")


def _init_converter() -> DocumentConverter:
    """Инициализация Docling конвертера (вызывается один раз при старте)."""
    return DocumentConverter()


def extract_files(uploaded_paths: list[Path], converter: DocumentConverter) -> list[ExtractedFile]:
    """
    Принимает список путей к загруженным файлам (могут быть ZIP).
    Возвращает список ExtractedFile с текстом.
    """
    results: list[ExtractedFile] = []

    # --- Шаг 1: инвентаризация ---
    all_files: list[Path] = []
    tmp_dirs: list[Path] = []

    for path in uploaded_paths:
        ext = path.suffix.lower()
        if ext == ".zip":
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_dirs.append(tmp_dir)
            unpacked = _unpack_zip(path, tmp_dir)
            all_files.extend(unpacked)
            logger.info(f"ZIP {path.name}: распаковано {len(unpacked)} файлов")
        else:
            all_files.append(path)

    # --- Шаг 2: извлечение текста через Docling ---
    for file_path in all_files:
        ext = file_path.suffix.lower()
        name = file_path.name

        if ext in SKIP_EXTENSIONS:
            logger.info(f"Пропущен {name}: формат {ext} не поддерживается")
            continue

        # Пропускаем служебные файлы из ZIP
        if any(part.startswith("__") or part.startswith(".") for part in file_path.parts):
            continue

        try:
            # .doc — через antiword
            if ext in ANTIWORD_EXTENSIONS:
                text = _extract_doc_antiword(file_path)
                results.append(ExtractedFile(name=name, path=str(file_path), text=text))
                logger.info(f"Извлечено (antiword) {name}: {len(text)} символов")

            # Остальные поддерживаемые форматы — через Docling
            elif ext in DOCLING_EXTENSIONS:
                result = converter.convert(str(file_path))
                text = result.document.export_to_markdown()
                if len(text.strip()) < 50:
                    logger.warning(f"{name}: текст слишком короткий ({len(text)} симв.)")
                results.append(ExtractedFile(name=name, path=str(file_path), text=text))
                logger.info(f"Извлечено (docling) {name}: {len(text)} символов")

            else:
                logger.info(f"Пропущен {name}: формат {ext} не поддерживается")

        except Exception as e:
            logger.error(f"Ошибка при обработке {name}: {e}")
            results.append(ExtractedFile(
                name=name, path=str(file_path), text="",
                skipped=True, skip_reason=f"Ошибка извлечения: {str(e)[:120]}"
            ))

    # Очистка временных директорий
    for tmp_dir in tmp_dirs:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results
