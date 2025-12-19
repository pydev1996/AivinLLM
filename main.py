from fastapi import FastAPI
from pydantic import BaseModel
import torch
from unsloth import FastLanguageModel
import boto3
import os

app = FastAPI(title="Qwen Interview Generator")

# -------------------------
# CONFIG
# -------------------------
BUCKET_NAME = "llm-prod-models"
S3_PREFIX = "Questions_finetuning_Qwen_model"
LOCAL_MODEL_DIR = "./model"

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

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=True,
        max_seq_length=2048,
    )

    # Load LoRA
    model.load_adapter(LOCAL_MODEL_DIR)

    FastLanguageModel.for_inference(model)

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

    messages = [
        {
            "role": "user",
            "content": f"Generate an interview question for the job role: {req.job_role}"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            num_return_sequences=req.num_outputs,
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
