from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import boto3
import os

app = FastAPI(title="Qwen Interview Generator (CPU)")

# -------------------------
# CONFIG
# -------------------------
BUCKET_NAME = "llm-prod-models"
S3_PREFIX = "Questions_finetuning_Qwen_model"
LOCAL_MODEL_DIR = "./lora"
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DEVICE = "cpu"

# -------------------------
# Load model on startup
# -------------------------
@app.on_event("startup")
def load_model():
    global model, tokenizer

    # Download LoRA from S3
    s3 = boto3.client("s3")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=S3_PREFIX
    )

    for obj in response.get("Contents", []):
        file_name = obj["Key"].split("/")[-1]
        if file_name:
            s3.download_file(
                BUCKET_NAME,
                obj["Key"],
                f"{LOCAL_MODEL_DIR}/{file_name}"
            )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True
    )

    # Base model (CPU)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        LOCAL_MODEL_DIR
    )

    model.to(DEVICE)
    model.eval()

# -------------------------
# Request schema
# -------------------------
class Request(BaseModel):
    job_role: str
    num_outputs: int = 5

# -------------------------
# API Endpoint
# -------------------------
@app.post("/generate")
def generate_questions(req: Request):

    # 🔒 UNCHANGED MESSAGE FORMAT
    messages = [
        {
            "role": "user",
            "content": f"Generate an interview question for the job role: {req.job_role}"
        }
    ]

    # Convert chat messages to text (Qwen style)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=req.num_outputs
        )

    results = []
    for out in outputs:
        decoded = tokenizer.decode(out, skip_special_tokens=True)
        answer = decoded.split("assistant")[-1].strip()
        results.append(answer)

    return {
        "job_role": req.job_role,
        "questions": results
    }
