from huggingface_hub import notebook_login
notebook_login()

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

lora_path = "amithsourya/Script-Generate-4GL-V1.0"
peft_config = PeftConfig.from_pretrained(lora_path)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(base_model, lora_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
prompt = "invoke a BO for read"
outputs = pipe(prompt, max_new_tokens=256)

print(outputs[0]["generated_text"])
