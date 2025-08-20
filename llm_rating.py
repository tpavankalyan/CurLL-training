'''python run_inference.py --model_path Pavankalyan/stage0_training --data_type instruct --split_type val --stage 0'''

import json
import ray
import logging
import os
import pickle
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
import argparse

# ========== HELPERS ==========
def load_json(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)

def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def log_error(e, idx, row):
    logging.error(f"Error at index {idx}: {e}\nRow: {row}")

if __name__ == "__main__":
    # ========== ARGS PARSE ==========
    parser = argparse.ArgumentParser(description="Run inference on prepared data using vLLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Name for the model column in Sheets.")
    parser.add_argument("--data_type", type=str, required=True, choices=["instruct", "cqa", "csqa"], help="Path or name of the HF dataset.")
    parser.add_argument("--split_type", type=str, default="val", choices=["val", "test"], help="Dataset split to use.")
    parser.add_argument("--stage", type=int, required=True, help="Stage number for the results.")
    args = parser.parse_args()
    
    stage = args.stage
    data_type = args.data_type
    split_type = args.split_type
    model_name = os.path.basename(args.model_path.rstrip('/'))
    model_name = model_name.replace('/', '_').replace('-', '_')
    
    # ========== CONFIGURATION ==========
    seed_data_path = f"/datadrive/pavan/az_storage/CL_results/seed/stage{stage}/{data_type}/{split_type}/{model_name}.pkl"
    prompt_path = f"/datadrive/pavan/az_storage/CL_results/{data_type}_prompt.json"
    output_path = f"/datadrive/pavan/az_storage/CL_results/outputs/stage{stage}/{data_type}/{split_type}/{model_name}"
    checkpoint_dir = f"/datadrive/pavan/az_storage/CL_results/outputs/stage{stage}/{data_type}/{split_type}/{model_name}"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========== LOGGING ==========
    logging.basicConfig(
        filename=os.path.join(checkpoint_dir, "processing.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    # ========== LOAD DATA ==========
    seed_data = load_data(seed_data_path)
    prompts = load_json(prompt_path)
    system_prompt = prompts.get('system', "You are a helpful assistant.")
    user_prompt_template = prompts['user']

    # ========== PREPARE PROMPTS ==========
    prepared_data = []
    for row in seed_data:
        try:
            user_text = user_prompt_template.format(**row)
        except Exception as e:
            logging.error(f"Prompt formatting error: {e} for row: {row}")
            user_text = ""
        prepared_data.append({
            "system": system_prompt,
            "user": user_text
        })

    # ========== INIT RAY ==========
    assert Version(ray.__version__) >= Version("2.44.1")
    ray.shutdown()  # Ensure no old cluster is connected
    ray.init(
        address=None,  # Explicitly says: "do not connect to any existing cluster"
        log_to_driver=False,
        local_mode=False  # or True for debugging, but False for real parallelism
    )

    # ========== RAY DATASET ==========
    ds = ray.data.from_items(prepared_data)

    config = vLLMEngineProcessorConfig(
        model_source="google/gemma-3-27b-it",
        engine_kwargs={
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 2048,
            "pipeline_parallel_size": 1,
        },
        concurrency=4,        # Increase concurrency for more throughput
        batch_size=256,        # Tune based on available resources
    )

    def preprocess(row):
        return dict(
            messages=[
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]},
            ],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=1024,
            ),
        )

    def postprocess(row):
        return dict(
            answer=row.get("generated_text", ""),
            **row,
        )

    vllm_processor = build_llm_processor(
        config,
        preprocess=preprocess,
        postprocess=postprocess,
    )

    # ========== RUN ==========
    try:
        processed_ds = vllm_processor(ds)
        processed_ds.write_parquet(output_path)  # output_path should be a directory!
        print(f"Processing complete. Output saved to {output_path} (as Parquet files)")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"Fatal error occurred. Check logs at {checkpoint_dir}/processing.log")