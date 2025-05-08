import streamlit as st
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


def generate_poem(topic):
    """
    Sends a POST request to the FastAPI server at http://localhost:8000/poem/invoke 
    to generate a poem on the provided topic.

    Args:
        topic (str): The topic for the poem.

    Returns:
        str: The generated poem content or an error message if the request fails.
    """
    response = requests.post("http://localhost:8000/poem/invoke",
                                json={'input':{'topic':topic}})
    if response.status_code == 200:
        return response.json()['output']
    else:
        return "Error: Unable to generate poem."
    
@st.cache_resource
def load_blip_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image):
    processor, model = load_blip_model()

    text = "A picture of"
    inputs = processor(images=image, text=text, return_tensors="pt")

    outputs = model(**inputs)
    print(outputs)
    return outputs

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.title("Генератор віршів за зображенням")

    st.write("Цей додаток дозволяє створити вірш на основі завантаженого зображення.")

    uploaded_file = st.file_uploader("Завантажте зображення", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        generate_image_caption(image)
        st.image(image, caption="Завантажене зображення", use_column_width=True)

        if st.button("Згенерувати вірш"):
            result = generate_poem("blank")
            st.subheader("Згенерований вірш")
            st.write(result)
    else:
        st.write("Будь ласка, завантажте зображення.")

if __name__ == "__main__":
  main()