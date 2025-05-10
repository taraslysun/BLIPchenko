from unsloth import FastLanguageModel
from fastapi import FastAPI, Query
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig, pipeline
from PIL import Image
import torch
from io import BytesIO
import base64
import deepl
import os
from dotenv import load_dotenv
from google import genai


from models.models import GenerateResponse, PoemResponse, ImageRequest, PoemRequest

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deepL_auth_key = os.getenv("DEEPL_AUTH_KEY")
deepL_client = deepl.DeepLClient(deepL_auth_key)
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Configure BLIP2 model in 8-bit
quant_config = BitsAndBytesConfig(load_in_8bit=True)
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b", use_fast=True
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.eval()

llama_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    device_map="auto",
)
adapter_path = f"out_jsonl"
llama_model.load_adapter(adapter_path)
FastLanguageModel.for_inference(llama_model)

app = FastAPI()


# THE GEMINI USAGE IS FOR REFERENCE AND RESEARCH PURPOSES ONLY
# THE MAIN GENERATION IS DONE WITH THE LLAMA MODEL
def generate_poem_llm(text: str, adapter_name: str, use_gemini: int) -> str:
    prompt = (
        f"Write a poem in style of Taras Shevchenko n Ukrainian: {text}"
    )
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    )

    print(response.text)


    # Create a fresh pipeline with the adapter loaded
    pipe = pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=True,
        top_k=50,
        temperature=0.8,
    )
    return pipe(text)[0]["generated_text"].strip(), response.text



@app.get("/", response_model=str)
def read_root():
    return "Welcome to the unified BLIP-2 /generate API!"



@app.post("/generate", response_model=GenerateResponse)
def generate(request: ImageRequest,
             adapter: str = Query("out_jsonl", enum=["out_jsonl", "out_raw", "out_small"]),
             use_gemini: bool = Query(False)):
    image_data = base64.b64decode(request.image_base64)
    image = Image.open(BytesIO(image_data)).convert("RGB")


    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)


    translation = deepL_client.translate_text(caption, target_lang="UK")
    translated_caption = translation.text


    poem, llm_ouput = generate_poem_llm(translated_caption, adapter, use_gemini)

    return GenerateResponse(
        caption=caption,
        translated_caption=translated_caption,
        poem=poem,
        llm_output=llm_ouput
    )

@app.post("/poem", response_model=PoemResponse)
def generate_poem(req: PoemRequest):
    """
    Generate a poem from arbitrary text, choosing adapter or gemini.
    """
    poem, llm_ouput = generate_poem_llm(req.text, req.adapter, req.use_gemini)
    return PoemResponse(poem=poem, llm_output=llm_ouput)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
