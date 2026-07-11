"""
This repo contains the FP8 version of Qwen3-4B-Thinking-2507, which has the following features:

Type: Causal Language Models
Training Stage: Pretraining & Post-training
Number of Parameters: 4.0B
Number of Paramaters (Non-Embedding): 3.6B
Number of Layers: 36
Number of Attention Heads (GQA): 32 for Q and 8 for KV
Context Length: 262,144 natively.
NOTE: This model supports only thinking mode. Meanwhile, specifying enable_thinking=True is no longer required.

Additionally, to enforce model thinking, the default chat template automatically includes <think>. Therefore, it is normal for the model's output to contain only </think> without an explicit opening <think> tag.
"""
import os
import sys
import argparse
import torch
import importlib.util

# =========================================================
# 1. ЖЕСТКИЙ ЗАПРЕТ НА СКАЧИВАНИЕ (ДО импорта transformers)
# =========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# =========================================================
# 2. ЗАГРУЗКА ЛОКАЛЬНОГО ЯДРА И ИНЪЕКЦИЯ В КЭШ TRANSFORMERS
# =========================================================
LOCAL_KERNEL_DIR = "/app/local_kernels_qwen3_8B_FP8"

def load_local_kernel_module():
    """Загружает папку как Python-модуль"""
    if not os.path.exists(LOCAL_KERNEL_DIR):
        raise FileNotFoundError(f"Локальное ядро не найдено по пути: {LOCAL_KERNEL_DIR}")
    
    init_file = os.path.join(LOCAL_KERNEL_DIR, "__init__.py")
    
    spec = importlib.util.spec_from_file_location(
        "finegrained_fp8_local", 
        init_file,
        submodule_search_locations=[LOCAL_KERNEL_DIR]
    )
    module = importlib.util.module_from_spec(spec)
    
    # Добавляем модуль в sys.modules, чтобы внутренние импорты внутри самого ядра сработали
    sys.modules["finegrained_fp8_local"] = module
    
    # Выполняем __init__.py (это подгрузит matmul.py, batched.py и т.д.)
    spec.loader.exec_module(module)
    return module

# Загружаем наш локальный модуль
local_fp8_kernel = load_local_kernel_module()

# ИМПОРТИРУЕМ МОДУЛЬ HUB_KERNELS И КЛАДЕМ НАШ МОДУЛЬ В ЕГО КЭШ
import transformers.integrations.hub_kernels as hub_kernels

# Инъекция! Когда lazy_load_kernel запросит "finegrained-fp8", 
# он увидит его в _KERNEL_MODULE_MAPPING и сразу вернет наш модуль.
hub_kernels._KERNEL_MODULE_MAPPING["finegrained-fp8"] = local_fp8_kernel

# =========================================================
# 3. ОСНОВНОЙ КОД
# =========================================================
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ckptdir',type=str,required=False,default='/mnt/nvme/huggingface')
    argparser.add_argument('--think',type=int,required=False,default=1)
    args = argparser.parse_args()
    model_path = f"{args.ckptdir}/models--Qwen--Qwen3-4B-Thinking-2507-FP8/snapshots/main"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True
    ).eval().to('cuda')

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content) # no opening <think> tag
    print("content:", content)
