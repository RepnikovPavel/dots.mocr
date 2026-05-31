from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
"""
pip install "kernels<0.15"
cache here:du -sh ~/.cache/huggingface/
"""
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ckptdir',type=str,required=False,default='/mnt/nvme/huggingface')
    args = argparser.parse_args()
    model_path = f"{args.ckptdir}/models--Qwen--Qwen3-8B-FP8/snapshots/main"
    model_name = "Qwen/Qwen3-8B-FP8"

    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True
    ).eval().to('cuda')

    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )

    with torch.no_grad():
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

    print("thinking content:", thinking_content)
    print("content:", content)
