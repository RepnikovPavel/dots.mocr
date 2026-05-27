import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from PIL import Image
from dots_mocr.utils import dict_promptmode_to_prompt

model_path = "/mnt/nvme/huggingface/models--rednote-hilab--dots.mocr/snapshots/fork"

# 1. Загрузка
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="sdpa",
    device_map="auto"
)
model.eval()

# 2. Подготовка входа
image_path = "/mnt/nvme/ocr_data/ko_2025/ko_2025_pages-to-jpg-0003.jpg"
messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": dict_promptmode_to_prompt['prompt_layout_all_en']}]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

print(f"1. Input IDs shape: {inputs.input_ids.shape}") # Длина текста + плейсхолдеры для картинки
print(f"2. Pixel Values shape: {inputs.pixel_values.shape}") # [Batch, Channels, Height, Width] (уже патчатый?)
print(f"3. Image Grid THW: {inputs.image_grid_thw}") # Временные и пространственные размерности

# 3. Прогон через Vision Tower (Ручками)
with torch.no_grad():
    # Достаем эмбеддинги текста
    text_embeds = model.get_input_embeddings()(inputs.input_ids)
    print(f"4. Text Embeddings shape: {text_embeds.shape}")
    
    # Прогоняем картинку через Vision Transformer
    img_mask = inputs.input_ids == model.config.image_token_id
    vision_embeds = model.vision_tower(inputs.pixel_values, inputs.image_grid_thw)
    print(f"5. Vision Embeddings shape (после ViT): {vision_embeds.shape}")
    
    # Merger (PatchMerger) уже внутри vision_tower сжимает эмбеддинги (spatial_merge_size=2 -> уменьшает в 4 раза)
    # Вставляем визуальные эмбеддинги в текстовые
    inputs_embeds = text_embeds.masked_scatter(
        img_mask.unsqueeze(-1).expand_as(text_embeds), vision_embeds
    )
    print(f"6. Fused Embeddings shape: {inputs_embeds.shape}") # Это подается в LLM

# 4. Генерация (Prefill + Decode)
with torch.no_grad():
    # Prefill (обработка промпта, генерация первого токена и KV Cache)
    out = model.generate(**inputs, max_new_tokens=10, return_dict_in_generate=True, output_scores=True)
    print(f"7. Сгенерированные токены: {out.sequences.shape}")