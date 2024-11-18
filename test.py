from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image



def generate_initial_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    image = Image.open(image_path)
    image = image.resize((512, 512))  # Use a larger image size
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, num_beams=5, no_repeat_ngram_size=2)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

generate_initial_caption("image.jpg")