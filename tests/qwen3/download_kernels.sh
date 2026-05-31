# Создаем изолированную папку и подпапку для внутренних импортов ядра
mkdir -p /app/local_kernels_qwen3_8B_FP8/finegrained_fp8

# Базовый URL репозитория
BASE_URL="https://huggingface.co/kernels-community/finegrained-fp8/resolve/main/build/torch-cuda"

# Скачиваем корневые файлы ядра
wget ${BASE_URL}/__init__.py -O /app/local_kernels_qwen3_8B_FP8/__init__.py
wget ${BASE_URL}/_ops.py -O /app/local_kernels_qwen3_8B_FP8/_ops.py
wget ${BASE_URL}/act_quant.py -O /app/local_kernels_qwen3_8B_FP8/act_quant.py
wget ${BASE_URL}/batched.py -O /app/local_kernels_qwen3_8B_FP8/batched.py
wget ${BASE_URL}/grouped.py -O /app/local_kernels_qwen3_8B_FP8/grouped.py
wget ${BASE_URL}/matmul.py -O /app/local_kernels_qwen3_8B_FP8/matmul.py
wget ${BASE_URL}/metadata.json -O /app/local_kernels_qwen3_8B_FP8/metadata.json
wget ${BASE_URL}/utils.py -O /app/local_kernels_qwen3_8B_FP8/utils.py

# Скачиваем файл из вложенной папки
wget ${BASE_URL}/finegrained_fp8/__init__.py -O /app/local_kernels_qwen3_8B_FP8/finegrained_fp8/__init__.py

echo "Ядра успешно скачаны в /app/local_kernels_qwen3_8B_FP8"