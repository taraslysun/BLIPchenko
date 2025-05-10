import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

FASTAPI_URL = "http://localhost:8000"

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_poem_from_image(image, adapter="out_jsonl", use_gemini=False):
    img_base64 = encode_image_to_base64(image)
    params = {
        "adapter": adapter,
        "use_gemini": 1
    }
    response = requests.post(
        f"{FASTAPI_URL}/generate",
        params=params,
        json={"image_base64": img_base64}
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API error: {response.status_code} - {response.text}")
        return None

def main():
    st.title("Poem generator")
    st.write("Upload image")

    adapter = st.selectbox("Оберіть адаптер", ["out_jsonl", "out_raw", "out_small"])
    use_gemini = st.checkbox("Use gemini")

    uploaded_file = st.file_uploader("Image(JPEG/PNG)", type=["jpg", "jpeg", "png"])
    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # convert to jpg
        if image.format != "JPEG":
            image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image = Image.open(buffered)
        st.image(image, caption="Uploaded image")

        if st.button("Generate poem"):
            with st.spinner("Generating"):
                result = generate_poem_from_image(image, adapter, use_gemini)
            if result:

                st.subheader("✍️ Generated with pretrained")
                #write with newlines 
                st.text(result['poem'])


                st.subheader("Other method")
                st.text(result['llm_output'])
    else:
        st.info("Upload image")

if __name__ == "__main__":
    main()
