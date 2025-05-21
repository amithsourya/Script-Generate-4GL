### Model Description

Generate 4GL Scripts from english prompts

- **Developed by:** Amith Sourya Sadineni
- **Model type:** Text Generation
- **Language(s):** Python
- **License:** MIT
- **Finetuned from model:** meta-llama/Llama-3.2-1B-Instruct

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://huggingface.co/amithsourya/Script-Generate-4GL-V1.0/blob/main/adapter_model.safetensors
- **Demo:**
```python
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
```

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** T4 GPU
- **Hours used:** 0H:23M

## Example


![image/png](https://cdn-uploads.huggingface.co/production/uploads/682b328fb814376780257a17/aaz_ESL50FOX-KLK4Xb8f.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/682b328fb814376780257a17/X4ZMrUHPVEy7-zI0rm5LL.png)
