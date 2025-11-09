
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
openai api fine_tunes.create \
  -t legal_qa_training.jsonl \
  -m gpt-3.5-turbo \
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
