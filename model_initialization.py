# thesis_project/models/test_models/test_initialization.py
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Qwen
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
print("Qwen loaded successfully!")

# Load Aya
aya_tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-23-8B", trust_remote_code=True)
aya_model = AutoModelForCausalLM.from_pretrained("CohereForAI/aya-23-8B", trust_remote_code=True)
print("Aya loaded successfully!")