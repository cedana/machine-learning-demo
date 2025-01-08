from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, BitsAndBytesConfig
from huggingface_hub import login
import torch

default_value = {
    'access_gated':False,
    'access_token':"",
    'model_id': "microsoft/phi-1_5", # "microsoft/Phi-3-mini-4k-instruct",
    'quantize':"bitsandbytes",
    'quant_level':"int8",
    'push_to_hub':False,
    'torch_device_map':"auto", 
    'torch_dtype':"auto", 
    'trust_remote_code':False, 
    'use_flash_attention_2':False, 
    'pipeline_task':"text-generation", 
    'max_new_tokens':500, 
    'return_full_text':False, 
    'temperature': 0.70,
    'do_sample': True, 
    'top_k':40, 
    'top_p':0.95, 
    'min_p':0.05, 
    'n_keep':0,
    'port':9069
}

## SET config above

model_id = default_value["model_id"]
model_params = {
    "device_map": default_value["torch_device_map"],
    "torch_dtype": default_value["torch_dtype"],
    "trust_remote_code": default_value["trust_remote_code"],
}
model = AutoModelForCausalLM.from_pretrained(model_id, **model_params)
tokenizer = AutoTokenizer.from_pretrained(model_id)

while True:
    user_input = "Explain what is a GPU?"

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    # Generate tokens
    tokens = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=default_value["temperature"],
        top_p=default_value["top_p"],
        do_sample=default_value["do_sample"],
    )

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f"\n---------------------------------------------------\nOutput:\n{output}", flush=True)
