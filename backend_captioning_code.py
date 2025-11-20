import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

!pip install transformers datasets torch torchvision pillow -q

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

print("🔄 Loading model...")

# Load model 
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("✅ Model loaded successfully!")

# Function to generate caption
def predict_caption(image_path):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Test with your image
caption = predict_caption("/kaggle/input/images/OIP (1).webp")
print("📝 Caption:", caption)
