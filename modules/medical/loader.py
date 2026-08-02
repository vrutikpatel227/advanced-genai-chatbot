"""
modules/medical/loader.py

Reusable MedQuAD dataset loading mechanism. Handles:
  - Automatic download of the dataset (from the configured URL --
    never hardcoded elsewhere) if not already present locally.
  - Parsing MedQuAD's XML schema into a flat list of MedicalQAPair.
  - Local JSON caching so the (relatively slow) XML parse only
    happens once; subsequent runs load the cache instantly.
  - Graceful handling of a missing dataset, download failure, and
    corrupted/unparseable individual XML files (skipped with a
    logged warning rather than aborting the whole load).

No dataset path is ever hardcoded in this file or elsewhere in the
module -- everything comes from modules/medical/config.py, which is
itself entirely env-driven.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from utils.logger import get_logger

from .config import MedicalConfig, medical_config

logger = get_logger(__name__)


class DatasetLoadError(Exception):
    """Raised when the MedQuAD dataset cannot be loaded, downloaded, or
    found anywhere. Callers (the RAG pipeline / UI) catch this and show
    a friendly message instead of letting the app crash."""


@dataclass(frozen=True)
class MedicalQAPair:
    doc_id: str
    pair_id: str
    question: str
    answer: str
    focus: str
    source: str
    url: str


class MedQuADLoader:
    """Loads the MedQuAD medical Q&A dataset: download (if needed) ->
    parse -> cache. Reusable for any future milestone that needs a
    similar "download a public dataset, parse it, cache it" pipeline."""

    def __init__(self, config: MedicalConfig | None = None) -> None:
        self._config = config or medical_config

    def load(self, force_refresh: bool = False) -> list[MedicalQAPair]:
        """Return the full list of parsed QA pairs, using the local
        cache when available. Raises DatasetLoadError if the dataset
        can't be obtained from any source (no cache, no local files,
        download fails) -- this is the "missing/corrupted dataset"
        error case from the PRD."""
        if not force_refresh and self._config.dataset_cache_path.exists():
            try:
                return self._load_from_cache()
            except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
                logger.warning(
                    "Cached MedQuAD dataset at %s is corrupted (%s); re-parsing from source.",
                    self._config.dataset_cache_path, exc,
                )

        if not self._has_raw_files():
            logger.info("No local MedQuAD files found; attempting automatic download.")
            self._download_and_extract()

        if not self._has_raw_files():
            raise DatasetLoadError(
                "The MedQuAD medical dataset is not available locally and could not "
                "be downloaded automatically. Set MEDICAL_DATASET_DIR to a local copy "
                "of the dataset, or check network access to GitHub."
            )

        pairs = self._parse_all()
        if not pairs:
            raise DatasetLoadError(
                "The MedQuAD dataset files were found but no valid Q&A pairs could "
                "be parsed from them -- the dataset may be corrupted."
            )

        self._save_to_cache(pairs)
        logger.info("Loaded and cached %d medical Q&A pairs.", len(pairs))
        return pairs

    # --- local file / cache helpers -------------------------------------------------

    def _has_raw_files(self) -> bool:
        if not self._config.dataset_dir.exists():
            return False
        return any(self._config.dataset_dir.rglob("*.xml"))

    def _load_from_cache(self) -> list[MedicalQAPair]:
        with open(self._config.dataset_cache_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        pairs = [MedicalQAPair(**item) for item in raw]
        logger.info("Loaded %d medical Q&A pairs from cache.", len(pairs))
        return pairs

    def _save_to_cache(self, pairs: list[MedicalQAPair]) -> None:
        self._config.dataset_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config.dataset_cache_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(p) for p in pairs], fh)

    # --- download ---------------------------------------------------------------

    def _download_and_extract(self) -> None:
        """Download the MedQuAD repo as a zip (single request, avoiding
        GitHub API rate limits) and extract its XML files. Never raises
        on failure -- logs and returns, so the caller's _has_raw_files()
        check surfaces a clean DatasetLoadError instead."""
        try:
            logger.info("Downloading MedQuAD dataset from %s ...", self._config.dataset_download_url)
            with urllib.request.urlopen(self._config.dataset_download_url, timeout=120) as response:
                zip_bytes = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.error("Failed to download MedQuAD dataset: %s", exc)
            return

        try:
            self._config.dataset_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                for member in archive.namelist():
                    if not member.endswith(".xml"):
                        continue
                    # Strip the top-level "MedQuAD-master/" directory from the path.
                    relative = Path(*Path(member).parts[1:]) if len(Path(member).parts) > 1 else Path(member)
                    target = self._config.dataset_dir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
            logger.info("MedQuAD dataset downloaded and extracted to %s", self._config.dataset_dir)
        except (zipfile.BadZipFile, OSError) as exc:
            logger.error("Failed to extract downloaded MedQuAD archive: %s", exc)

    # --- XML parsing --------------------------------------------------------------

    def _parse_all(self) -> list[MedicalQAPair]:
        xml_files = self._select_files()

        pairs: list[MedicalQAPair] = []
        skipped = 0
        for xml_path in xml_files:
            try:
                pairs.extend(self._parse_file(xml_path))
            except ET.ParseError as exc:
                skipped += 1
                logger.warning("Skipping corrupted MedQuAD file %s: %s", xml_path, exc)
            except OSError as exc:
                skipped += 1
                logger.warning("Skipping unreadable MedQuAD file %s: %s", xml_path, exc)

        if skipped:
            logger.warning("Skipped %d corrupted/unreadable files out of %d.", skipped, len(xml_files))
        return pairs

    def _select_files(self) -> list[Path]:
        """Pick which XML files to parse. When capped (max_source_files
        > 0), samples round-robin across each top-level source folder
        rather than taking the first N alphabetically -- MedQuAD's
        folders vary widely in content (e.g. the ADAM folder's answers
        are blanked out for copyright reasons), so a naive alphabetical
        slice can land entirely inside a low-content folder."""
        all_files = sorted(self._config.dataset_dir.rglob("*.xml"))
        if self._config.max_source_files <= 0:
            return all_files

        by_folder: dict[str, list[Path]] = {}
        for path in all_files:
            folder = path.relative_to(self._config.dataset_dir).parts[0]
            by_folder.setdefault(folder, []).append(path)

        selected: list[Path] = []
        folder_queues = [files for _, files in sorted(by_folder.items())]
        index = 0
        while len(selected) < self._config.max_source_files and any(folder_queues):
            queue = folder_queues[index % len(folder_queues)]
            if queue:
                selected.append(queue.pop(0))
            index += 1
            if index > 10_000_000:  # pathological safety valve, never expected to trigger
                break
        return selected

    @staticmethod
    def _parse_file(xml_path: Path) -> list[MedicalQAPair]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        doc_id = root.get("id", xml_path.stem)
        source = root.get("source", "MedQuAD")
        url = root.get("url", "")
        focus_el = root.find("Focus")
        focus = (focus_el.text or "").strip() if focus_el is not None else ""

        results: list[MedicalQAPair] = []
        for qa_pair in root.findall(".//QAPair"):
            question_el = qa_pair.find("Question")
            answer_el = qa_pair.find("Answer")
            if question_el is None or answer_el is None:
                continue
            question = (question_el.text or "").strip()
            answer = (answer_el.text or "").strip()
            if not question or not answer:
                continue
            pair_id = question_el.get("qid", f"{doc_id}-{len(results) + 1}")
            results.append(
                MedicalQAPair(
                    doc_id=doc_id,
                    pair_id=pair_id,
                    question=question,
                    answer=answer,
                    focus=focus,
                    source=source,
                    url=url,
                )
            )
        return results
