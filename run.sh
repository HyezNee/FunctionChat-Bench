# run singlecall evaluation
python3 evaluate.py singlecall \
--input_path data/FunctionChat-Singlecall.jsonl \
--tools_type all \
--system_prompt_path data/system_prompt_5.txt \
--temperature 1.0 \
--model Qwen3-8B \
--api_key your-api-key \
--model_path Qwen/Qwen3-8B

# run dialog evaluation
# python3 evaluate.py dialog \
# --input_path data/FunctionChat-Dialog.jsonl \
# --system_prompt_path data/system_prompt.txt \
# --temperature 0.1 \
# --model {model_name} \
# --api_key {api_key} 