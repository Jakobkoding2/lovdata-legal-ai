#!/usr/bin/env python3
"""
Legal Chat Model Training
Fine-tunes an instruction model for Norwegian legal Q&A using the OpenAI API.
Due to resource constraints, this creates a training dataset and provides instructions
for fine-tuning with external services.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import random

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"


class LegalQADatasetGenerator:
    """Generates Q&A pairs for legal chat model training"""
    
    def __init__(self, corpus_df: pd.DataFrame, similarity_df: pd.DataFrame):
        self.corpus_df = corpus_df
        self.similarity_df = similarity_df
    
    def generate_qa_pairs(self, num_samples: int = 1000) -> List[Dict]:
        """Generate Q&A pairs from legal texts"""
        
        print(f"\nGenerating {num_samples} Q&A pairs...")
        
        qa_pairs = []
        
        # Sample texts
        sample_indices = random.sample(range(len(self.corpus_df)), min(num_samples, len(self.corpus_df)))
        
        for idx in sample_indices:
            row = self.corpus_df.iloc[idx]
            
            # Generate different types of questions
            qa_pairs.extend(self._generate_questions_for_text(row))
        
        print(f"✓ Generated {len(qa_pairs)} Q&A pairs")
        
        return qa_pairs
    
    def _generate_questions_for_text(self, row: pd.Series) -> List[Dict]:
        """Generate multiple question types for a single legal text"""
        
        text = row['text_clean']
        doc_title = row['doc_title']
        section_num = row.get('section_num', 'N/A')
        group = row['group']
        
        qa_pairs = []
        
        # Type 1: Direct content question
        qa_pairs.append({
            'messages': [
                {'role': 'system', 'content': 'Du er en ekspert på norsk lov. Svar nøyaktig basert på lovteksten.'},
                {'role': 'user', 'content': f'Hva sier {doc_title} § {section_num} om dette?'},
                {'role': 'assistant', 'content': text}
            ],
            'metadata': {
                'doc_id': row['doc_id'],
                'group': group,
                'type': 'direct_content'
            }
        })
        
        # Type 2: Summary question
        summary = text[:200] + '...' if len(text) > 200 else text
        qa_pairs.append({
            'messages': [
                {'role': 'system', 'content': 'Du er en ekspert på norsk lov. Gi korte, presise svar.'},
                {'role': 'user', 'content': f'Oppsummer kort hva {doc_title} § {section_num} handler om.'},
                {'role': 'assistant', 'content': summary}
            ],
            'metadata': {
                'doc_id': row['doc_id'],
                'group': group,
                'type': 'summary'
            }
        })
        
        # Type 3: Semantic search question
        keywords = ' '.join(text.split()[:10])
        qa_pairs.append({
            'messages': [
                {'role': 'system', 'content': 'Du er en ekspert på norsk lov. Finn relevant lovtekst.'},
                {'role': 'user', 'content': f'Hvilke lover gjelder for {keywords}?'},
                {'role': 'assistant', 'content': f'I henhold til {doc_title} § {section_num}: {text}'}
            ],
            'metadata': {
                'doc_id': row['doc_id'],
                'group': group,
                'type': 'semantic_search'
            }
        })
        
        return qa_pairs
    
    def generate_overlap_qa(self, num_samples: int = 500) -> List[Dict]:
        """Generate Q&A pairs about overlaps and conflicts"""
        
        print(f"\nGenerating {num_samples} overlap Q&A pairs...")
        
        qa_pairs = []
        
        # Sample similarity pairs
        sample_pairs = self.similarity_df.sample(min(num_samples, len(self.similarity_df)))
        
        for _, pair in sample_pairs.iterrows():
            idx1, idx2 = pair['idx1'], pair['idx2']
            similarity = pair['similarity']
            
            row1 = self.corpus_df.iloc[idx1]
            row2 = self.corpus_df.iloc[idx2]
            
            # Generate overlap question
            if similarity > 0.95:
                question = f"Er det overlapp mellom {row1['doc_title']} og {row2['doc_title']}?"
                answer = f"Ja, det er høy grad av overlapp (similarity: {similarity:.2f}). Dette kan indikere duplikasjon eller subsumpsjon."
            elif similarity > 0.85:
                question = f"Hvordan forholder {row1['doc_title']} seg til {row2['doc_title']}?"
                answer = f"Det er betydelig semantisk likhet (similarity: {similarity:.2f}), som kan indikere delegasjon eller relaterte bestemmelser."
            else:
                question = f"Er {row1['doc_title']} og {row2['doc_title']} relatert?"
                answer = f"Det er moderat semantisk likhet (similarity: {similarity:.2f}), men de behandler sannsynligvis forskjellige aspekter."
            
            qa_pairs.append({
                'messages': [
                    {'role': 'system', 'content': 'Du er en ekspert på norsk lov. Analyser forhold mellom lover.'},
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer}
                ],
                'metadata': {
                    'doc1_id': row1['doc_id'],
                    'doc2_id': row2['doc_id'],
                    'similarity': float(similarity),
                    'type': 'overlap_analysis'
                }
            })
        
        print(f"✓ Generated {len(qa_pairs)} overlap Q&A pairs")
        
        return qa_pairs
    
    def save_training_data(self, qa_pairs: List[Dict], filepath: Path):
        """Save training data in OpenAI fine-tuning format (JSONL)"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                # OpenAI fine-tuning format
                training_example = {
                    'messages': qa['messages']
                }
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Saved training data to {filepath} ({size_mb:.2f} MB)")
        
        # Also save with metadata for analysis
        metadata_path = filepath.parent / (filepath.stem + '_with_metadata.jsonl')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved metadata version to {metadata_path}")


def create_training_instructions():
    """Create instructions for fine-tuning"""
    
    instructions = """
# Legal Chat Model Fine-Tuning Instructions

## Overview
This dataset contains Norwegian legal Q&A pairs for fine-tuning instruction models.

## Dataset Statistics
- Total Q&A pairs: See training file
- Question types: direct_content, summary, semantic_search, overlap_analysis
- Languages: Norwegian (Bokmål)

## Recommended Models for Fine-Tuning

### Option 1: OpenAI GPT-4 / GPT-3.5
```bash
# Upload training file
openai api fine_tunes.create \\
  -t legal_qa_training.jsonl \\
  -m gpt-3.5-turbo \\
  --suffix "norwegian-legal"

# Monitor training
openai api fine_tunes.follow -i <fine_tune_id>
```

### Option 2: Hugging Face Models
Recommended base models:
- `NbAiLab/nb-gpt-j-6B` (Norwegian GPT)
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-3-8B-Instruct`

Use the Hugging Face Trainer API or PEFT/LoRA for efficient fine-tuning.

### Option 3: Local Fine-Tuning with LoRA
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load base model
model = AutoModelForCausalLM.from_pretrained("NbAiLab/nb-gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-gpt-j-6B")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Train with your dataset
# ... (use Trainer API)
```

## Performance Targets
- Semantic QA accuracy: ≥ 80% F1 vs baseline GPT-4-Turbo
- Response latency: < 2s per query
- Training time: < 3h on single GPU

## Deployment
After fine-tuning, deploy the model using:
1. Hugging Face Inference API
2. Local FastAPI server (see api_server.py)
3. Cloud deployment (AWS SageMaker, Azure ML, GCP Vertex AI)

## Notes
- The dataset is auto-generated from legal texts
- Consider human review for production use
- Compliance with Norwegian data protection laws required
"""
    
    return instructions


def main():
    """Main training data generation pipeline"""
    print("=" * 60)
    print("Legal Chat Model Training Data Generation")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    corpus_path = PROCESSED_DIR / "lovdata_corpus.parquet"
    similarity_path = PROCESSED_DIR / "similarity_pairs.parquet"
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        sys.exit(1)
    
    corpus_df = pd.read_parquet(corpus_path)
    
    # Limit to subset used for embeddings
    corpus_df = corpus_df.head(20000)
    
    similarity_df = pd.DataFrame()
    if similarity_path.exists():
        similarity_df = pd.read_parquet(similarity_path)
    
    print(f"✓ Loaded {len(corpus_df)} corpus texts")
    print(f"✓ Loaded {len(similarity_df)} similarity pairs")
    
    # Generate Q&A pairs
    print("\n[2/4] Generating Q&A pairs...")
    generator = LegalQADatasetGenerator(corpus_df, similarity_df)
    
    # Generate content Q&A
    content_qa = generator.generate_qa_pairs(num_samples=500)
    
    # Generate overlap Q&A
    overlap_qa = []
    if len(similarity_df) > 0:
        overlap_qa = generator.generate_overlap_qa(num_samples=200)
    
    all_qa = content_qa + overlap_qa
    
    print(f"\n✓ Total Q&A pairs: {len(all_qa)}")
    
    # Save training data
    print("\n[3/4] Saving training data...")
    training_path = PROCESSED_DIR / "legal_qa_training.jsonl"
    generator.save_training_data(all_qa, training_path)
    
    # Create instructions
    print("\n[4/4] Creating training instructions...")
    instructions = create_training_instructions()
    instructions_path = MODELS_DIR / "FINE_TUNING_INSTRUCTIONS.md"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    print(f"✓ Saved instructions to {instructions_path}")
    
    # Create statistics
    stats = {
        'total_qa_pairs': len(all_qa),
        'content_qa_pairs': len(content_qa),
        'overlap_qa_pairs': len(overlap_qa),
        'unique_documents': int(corpus_df['doc_id'].nunique()),
        'question_types': {
            'direct_content': len([qa for qa in content_qa if qa['metadata']['type'] == 'direct_content']),
            'summary': len([qa for qa in content_qa if qa['metadata']['type'] == 'summary']),
            'semantic_search': len([qa for qa in content_qa if qa['metadata']['type'] == 'semantic_search']),
            'overlap_analysis': len(overlap_qa)
        }
    }
    
    stats_path = MODELS_DIR / "chat_training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Data Generation Complete!")
    print("=" * 60)
    print(f"Total Q&A pairs: {stats['total_qa_pairs']}")
    print(f"Content Q&A: {stats['content_qa_pairs']}")
    print(f"Overlap Q&A: {stats['overlap_qa_pairs']}")
    print(f"Training file: {training_path}")
    print(f"Instructions: {instructions_path}")
    print("=" * 60)
    print("\nNOTE: Due to resource constraints, actual model fine-tuning")
    print("should be done using OpenAI API or Hugging Face services.")
    print("See FINE_TUNING_INSTRUCTIONS.md for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
