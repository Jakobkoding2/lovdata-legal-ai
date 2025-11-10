from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .config import (
    BASE_DIR,
    CORPUS_PATH,
    DEFAULT_LAW_SOURCE_URL,
    PROCESSED_DIR,
    RAW_DIR,
)
from .logging_utils import get_logger

logger = get_logger("lovdata_rag.data")

LAWS_URL = "https://api.lovdata.no/v1/publicData/get/gjeldende-lover.tar.bz2"
REGS_URL = "https://api.lovdata.no/v1/publicData/get/gjeldende-sentrale-forskrifter.tar.bz2"


@dataclass
class CorpusRecord:
    id: str
    doc_id: str
    law_id: str
    law_name: str
    doc_title: str
    kapittel: Optional[str]
    paragraf: Optional[str]
    ledd: Optional[str]
    text: str
    date: Optional[str]
    source_url: str
    group: str
    raw_path: str


class LovdataFetcher:
    def __init__(self, target_dir: Path = RAW_DIR):
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, filename: str, force: bool = False) -> Path:
        destination = self.target_dir / filename
        if destination.exists() and not force:
            logger.info("Dataset %s already downloaded", filename)
            return destination
        logger.info("Downloading %s", url)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"download:{filename}"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return destination

    def extract(self, archive_path: Path, force: bool = False) -> Path:
        extract_dir = archive_path.with_suffix("")
        if extract_dir.exists() and not force:
            logger.info("Archive %s already extracted", archive_path.name)
            return extract_dir
        logger.info("Extracting %s", archive_path.name)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=extract_dir)
        return extract_dir

    def fetch_all(self, force: bool = False) -> Dict[str, Path]:
        laws = self.extract(self.download(LAWS_URL, "gjeldende-lover.tar.bz2", force=force), force=force)
        regs = self.extract(
            self.download(REGS_URL, "gjeldende-sentrale-forskrifter.tar.bz2", force=force), force=force
        )
        return {"law": laws, "regulation": regs}


class LovdataParser:
    def __init__(self) -> None:
        pass

    def parse_directory(self, directory: Path, group: str) -> Iterable[CorpusRecord]:
        xml_files = sorted(directory.rglob("*.xml"))
        for path in tqdm(xml_files, desc=f"parse:{group}"):
            yield from self.parse_file(path, group)

    def parse_file(self, path: Path, group: str) -> List[CorpusRecord]:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")
        soup = BeautifulSoup(content, "html.parser")

        law_id = path.stem
        law_name = self._extract_title(soup) or "Ukjent lov"
        kapittel = None

        records: List[CorpusRecord] = []

        for article in soup.find_all("article", class_="legalArticle"):
            kapittel = self._get_chapter(article, kapittel)
            paragraf = self._get_paragraph(article)
            ledd_sections = article.find_all("article", class_="legalP")
            if not ledd_sections:
                text = article.get_text(separator=" ", strip=True)
                if text:
                    records.append(
                        self._record(
                            law_id,
                            law_name,
                            kapittel,
                            paragraf,
                            None,
                            text,
                            group,
                            path,
                            article,
                        )
                    )
                continue

            for idx, ledd in enumerate(ledd_sections, start=1):
                text = ledd.get_text(separator=" ", strip=True)
                if not text:
                    continue
                records.append(
                    self._record(
                        law_id,
                        law_name,
                        kapittel,
                        paragraf,
                        f"{paragraf or '§'}-{idx}",
                        text,
                        group,
                        path,
                        ledd,
                    )
                )
        return records

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        if soup.title and soup.title.text:
            return soup.title.text.strip()
        h1 = soup.find("h1")
        if h1 and h1.text:
            return h1.text.strip()
        return None

    def _get_chapter(self, article, current: Optional[str]) -> Optional[str]:
        chapter_tag = article.find("span", class_="title")
        if chapter_tag:
            return chapter_tag.get_text(strip=True)
        return current

    def _get_paragraph(self, article) -> Optional[str]:
        span = article.find("span", class_="legalArticleValue")
        if span:
            return span.get_text(strip=True)
        name = article.attrs.get("data-name")
        if name:
            return name.strip()
        return None

    def _record(
        self,
        law_id: str,
        law_name: str,
        kapittel: Optional[str],
        paragraf: Optional[str],
        ledd: Optional[str],
        text: str,
        group: str,
        path: Path,
        tag,
    ) -> CorpusRecord:
        attr_date = tag.attrs.get("data-date") or None
        source_url = tag.attrs.get("data-url") or DEFAULT_LAW_SOURCE_URL
        unique = hashlib.md5(f"{law_id}:{paragraf}:{ledd}:{len(text)}".encode("utf-8")).hexdigest()
        return CorpusRecord(
            id=f"{law_id}_{unique}",
            doc_id=law_id,
            law_id=law_id,
            law_name=law_name,
            doc_title=law_name,
            kapittel=kapittel,
            paragraf=paragraf,
            ledd=ledd,
            text=text,
            date=attr_date,
            source_url=source_url,
            group=group,
            raw_path=str(path.relative_to(BASE_DIR)),
        )


def build_corpus(force: bool = False) -> Path:
    if CORPUS_PATH.exists() and not force:
        logger.info("Corpus already exists at %s", CORPUS_PATH)
        return CORPUS_PATH

    fetcher = LovdataFetcher()
    directories = fetcher.fetch_all(force=force)
    parser = LovdataParser()

    records: List[Dict] = []
    for group, folder in directories.items():
        for record in parser.parse_directory(folder, group):
            records.append(asdict(record))

    if not records:
        raise RuntimeError("No legal documents parsed from Lovdata dumps")

    df = pd.DataFrame(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CORPUS_PATH, index=False)

    stats = {
        "total_rows": len(df),
        "laws": int(df[df["group"] == "law"]["law_id"].nunique()),
        "regulations": int(df[df["group"] == "regulation"]["law_id"].nunique()),
    }
    stats_path = PROCESSED_DIR / "lovdata_corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Saved corpus with %s rows to %s", len(df), CORPUS_PATH)
    return CORPUS_PATH
