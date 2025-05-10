from pydantic import BaseModel
from fastapi import Query

class ImageRequest(BaseModel):
    image_base64: str

class PoemRequest(BaseModel):
    text: str
    adapter: str = Query("out_jsonl", enum=["out_jsonl", "out_raw", "out_small"])
    use_gemini: bool = False

class GenerateResponse(BaseModel):
    caption: str
    translated_caption: str
    poem: str
    llm_output: str

class PoemResponse(BaseModel):
    poem: str
    llm_output: str