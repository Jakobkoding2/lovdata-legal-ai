#!/usr/bin/env python3
"""
Data Pipeline for Lovdata Legal Texts
Fetches, extracts, parses, and normalizes Norwegian laws and regulations.
"""

import os
import sys
import json
import tarfile
import requests
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

# Configuration
LOVDATA_LAWS_URL = "https://api.lovdata.no/v1/publicData/get/gjeldende-lover.tar.bz2"
LOVDATA_REGS_URL = "https://api.lovdata.no/v1/publicData/get/gjeldende-sentrale-forskrifter.tar.bz2"

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LegalUnit:
    """Represents a single legal text unit (paragraph/section)"""
    unit_id: str
    doc_id: str
    doc_title: str
    group: str  # 'law' or 'regulation'
    section_num: Optional[str]
    section_title: Optional[str]
    text_raw: str
    text_clean: str
    char_count: int
    word_count: int


class LovdataFetcher:
    """Handles downloading and extraction of Lovdata datasets"""
    
    def __init__(self, data_dir: Path = RAW_DIR):
        self.data_dir = data_dir
        
    def download_dataset(self, url: str, filename: str) -> Path:
        """Download a dataset from Lovdata API"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} already exists, skipping download")
            return filepath
            
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        print(f"✓ Downloaded {filename}")
        return filepath
    
    def extract_archive(self, archive_path: Path) -> Path:
        """Extract tar.bz2 archive"""
        extract_dir = self.data_dir / archive_path.stem.replace('.tar', '')
        
        if extract_dir.exists():
            print(f"✓ {extract_dir.name} already extracted")
            return extract_dir
            
        print(f"Extracting {archive_path.name}...")
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(path=extract_dir)
            
        print(f"✓ Extracted to {extract_dir}")
        return extract_dir
    
    def fetch_all(self) -> Dict[str, Path]:
        """Download and extract both laws and regulations"""
        laws_archive = self.download_dataset(LOVDATA_LAWS_URL, "gjeldende-lover.tar.bz2")
        regs_archive = self.download_dataset(LOVDATA_REGS_URL, "gjeldende-sentrale-forskrifter.tar.bz2")
        
        laws_dir = self.extract_archive(laws_archive)
        regs_dir = self.extract_archive(regs_archive)
        
        return {
            'laws': laws_dir,
            'regulations': regs_dir
        }


class LegalTextParser:
    """Parses HTML/XML legal documents into structured units"""
    
    # Boilerplate patterns to remove
    BOILERPLATE_PATTERNS = [
        r'opphevet ved',
        r'endret ved',
        r'tilføyd ved',
        r'fastsatt av',
        r'med hjemmel i',
        r'i kraft fra',
        r'trådt i kraft'
    ]
    
    def __init__(self):
        self.boilerplate_regex = re.compile(
            '|'.join(self.BOILERPLATE_PATTERNS),
            re.IGNORECASE
        )
    
    def parse_xml_file(self, filepath: Path, group: str) -> List[LegalUnit]:
        """Parse a single HTML/XML file into legal units"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []
        
        soup = BeautifulSoup(content, 'html.parser')
        units = []
        
        # Extract document metadata
        doc_id = filepath.stem
        doc_title = self._extract_title(soup)
        
        # Find all legal articles and paragraphs
        # Look for <article class="legalArticle"> and <article class="legalP">
        articles = soup.find_all('article', class_='legalArticle')
        
        for article in articles:
            # Extract section number
            section_num = self._extract_section_num(article)
            section_title = None  # Most don't have titles
            
            # Find all paragraphs within this article
            paragraphs = article.find_all('article', class_='legalP')
            
            for idx, para in enumerate(paragraphs):
                text_raw = para.get_text(separator=' ', strip=True)
                
                if not text_raw or len(text_raw) < 20:
                    continue
                
                # Clean text
                text_clean = self._clean_text(text_raw)
                
                # Skip if mostly boilerplate
                if self._is_boilerplate(text_clean):
                    continue
                
                para_id = para.get('id', f'para_{idx}')
                
                unit = LegalUnit(
                    unit_id=f"{doc_id}_{para_id}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    group=group,
                    section_num=section_num,
                    section_title=section_title,
                    text_raw=text_raw,
                    text_clean=text_clean,
                    char_count=len(text_clean),
                    word_count=len(text_clean.split())
                )
                units.append(unit)
        
        # Also check for standalone paragraphs not in articles
        standalone_paras = soup.find_all('article', class_='legalP')
        for idx, para in enumerate(standalone_paras):
            # Skip if already processed as part of an article
            if para.find_parent('article', class_='legalArticle'):
                continue
                
            text_raw = para.get_text(separator=' ', strip=True)
            
            if not text_raw or len(text_raw) < 20:
                continue
            
            text_clean = self._clean_text(text_raw)
            
            if self._is_boilerplate(text_clean):
                continue
            
            para_id = para.get('id', f'standalone_{idx}')
            
            unit = LegalUnit(
                unit_id=f"{doc_id}_{para_id}",
                doc_id=doc_id,
                doc_title=doc_title,
                group=group,
                section_num=None,
                section_title=None,
                text_raw=text_raw,
                text_clean=text_clean,
                char_count=len(text_clean),
                word_count=len(text_clean.split())
            )
            units.append(unit)
        
        return units
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract document title"""
        # Try h1 first (main title)
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        # Try title tag
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)
        
        return "Untitled"
    
    def _extract_section_num(self, article) -> Optional[str]:
        """Extract section/paragraph number from article"""
        # Look for span with class legalArticleValue
        value_span = article.find('span', class_='legalArticleValue')
        if value_span:
            return value_span.get_text(strip=True)
        
        # Try data-name attribute
        if article.has_attr('data-name'):
            return article['data-name']
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Norwegian letters and common punctuation
        text = re.sub(r'[^\w\s\.,;:!?()\-§«»æøåÆØÅ]', '', text)
        
        # Normalize punctuation spacing
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        return text.strip()
    
    def _is_boilerplate(self, text: str) -> bool:
        """Check if text is primarily boilerplate"""
        if len(text) < 30:
            return True
        
        # Check for boilerplate patterns
        matches = self.boilerplate_regex.findall(text.lower())
        
        # If more than 40% is boilerplate, skip
        boilerplate_chars = sum(len(m) for m in matches)
        if len(text) > 0 and (boilerplate_chars / len(text)) > 0.4:
            return True
        
        # Skip if it's just metadata
        metadata_indicators = ['opphevet', 'endret', 'tilføyd', 'i kraft']
        if any(ind in text.lower() for ind in metadata_indicators) and len(text) < 100:
            return True
        
        return False
    
    def parse_directory(self, directory: Path, group: str) -> List[LegalUnit]:
        """Parse all XML files in a directory"""
        xml_files = list(directory.rglob('*.xml'))
        
        print(f"\nParsing {len(xml_files)} {group} files...")
        
        all_units = []
        for filepath in tqdm(xml_files):
            units = self.parse_xml_file(filepath, group)
            all_units.extend(units)
        
        print(f"✓ Extracted {len(all_units)} units from {group}")
        return all_units


class DatasetBuilder:
    """Builds and exports processed datasets"""
    
    def __init__(self, output_dir: Path = PROCESSED_DIR):
        self.output_dir = output_dir
    
    def build_dataset(self, units: List[LegalUnit]) -> pd.DataFrame:
        """Convert legal units to DataFrame"""
        if not units:
            print("WARNING: No units to build dataset from!")
            return pd.DataFrame()
        
        data = [asdict(unit) for unit in units]
        df = pd.DataFrame(data)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Save dataset in multiple formats"""
        if df.empty:
            print("ERROR: Cannot save empty dataset")
            return {}
        
        # Save as Parquet (efficient)
        parquet_path = self.output_dir / f"{filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"✓ Saved {parquet_path}")
        
        # Save as JSONL (for Hugging Face)
        jsonl_path = self.output_dir / f"{filename}.jsonl"
        df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
        print(f"✓ Saved {jsonl_path}")
        
        # Save summary stats
        stats = {
            'total_units': len(df),
            'groups': df['group'].value_counts().to_dict(),
            'avg_text_length': float(df['text_clean'].str.len().mean()),
            'total_documents': int(df['doc_id'].nunique()),
            'avg_words_per_unit': float(df['word_count'].mean())
        }
        
        stats_path = self.output_dir / f"{filename}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved {stats_path}")
        
        return stats


def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("Lovdata Legal Data Pipeline")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n[1/3] Fetching datasets...")
    fetcher = LovdataFetcher()
    directories = fetcher.fetch_all()
    
    # Step 2: Parse data
    print("\n[2/3] Parsing legal texts...")
    parser = LegalTextParser()
    
    laws_units = parser.parse_directory(directories['laws'], 'law')
    regs_units = parser.parse_directory(directories['regulations'], 'regulation')
    
    all_units = laws_units + regs_units
    
    if not all_units:
        print("\nERROR: No legal units were extracted!")
        print("Please check the XML file format and parsing logic.")
        sys.exit(1)
    
    # Step 3: Build and save dataset
    print("\n[3/3] Building dataset...")
    builder = DatasetBuilder()
    df = builder.build_dataset(all_units)
    stats = builder.save_dataset(df, 'lovdata_corpus')
    
    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total units: {stats['total_units']}")
    print(f"Laws: {stats['groups'].get('law', 0)}")
    print(f"Regulations: {stats['groups'].get('regulation', 0)}")
    print(f"Avg text length: {stats['avg_text_length']:.0f} chars")
    print(f"Avg words per unit: {stats['avg_words_per_unit']:.1f}")
    print(f"Total documents: {stats['total_documents']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
