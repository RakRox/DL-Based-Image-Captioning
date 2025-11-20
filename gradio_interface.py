import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

import logging
logging.getLogger().setLevel(logging.ERROR)

!pip install gradio transformers torch pillow -q > /dev/null 2>&1

import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

print("🔄 Setting up AI Caption Generator...")

# Load model quietly
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("✅ AI Caption Generator is ready!")

def generate_caption(image):
    """Generate caption for uploaded image"""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
        
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    except Exception as e:
        return "Sorry, I couldn't generate a caption for this image."

def generate_caption_from_url(url):
    """Generate caption from image URL"""
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        return generate_caption(image)
    except:
        return "Please check your internet connection or try a different image URL."

# Create clean Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Image Caption Generator") as demo:
    
    gr.Markdown(
        """
        <div style="text-align: center;">
        <h1>🖼 Image Caption Generator</h1>
        <p>Upload any image and get an automatic caption instantly!</p>
        </div>
        """
    )
    
    with gr.Tab("📸 Upload Image"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: Upload your image")
                image_input = gr.Image(
                    label="",
                    type="pil",
                    height=300,
                    show_label=False
                )
                upload_btn = gr.Button("✨ Generate Caption", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### Step 2: Your generated caption")
                caption_output = gr.Textbox(
                    label="",
                    placeholder="Your image caption will appear here...",
                    lines=4,
                    show_label=False
                )
        
        gr.Markdown("Try these example images:")
        gr.Examples(
            examples=[
                ["https://images.unsplash.com/photo-1541963463532-d68292c34b19?w=400"],
                ["https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400"],
                ["https://images.unsplash.com/photo-1575936123452-b67c3203c357?w=400"]
            ],
            inputs=image_input,
            outputs=caption_output,
            fn=generate_caption,
            cache_examples=True,
            label=""
        )
    
    with gr.Tab("🔗 From Web URL"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Paste image URL from web")
                url_input = gr.Textbox(
                    label="",
                    placeholder="Paste image link here...",
                    lines=2,
                    show_label=False
                )
                url_btn = gr.Button("🌐 Generate Caption", variant="primary")
            
            with gr.Column():
                url_output = gr.Textbox(
                    label="",
                    placeholder="Caption from web image...",
                    lines=4,
                    show_label=False
                )
    
    # Footer
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #666;">
        <p>💡 <strong>Tips for better captions:</strong> Use clear, well-lit photos with recognizable objects and scenes.</p>
        <p>Perfect for: Nature photos • People • Animals • Objects • Everyday scenes</p>
        </div>
        """
    )
    
    # Event handlers
    upload_btn.click(
        fn=generate_caption,
        inputs=image_input,
        outputs=caption_output
    )
    
    url_btn.click(
        fn=generate_caption_from_url,
        inputs=url_input,
        outputs=url_output
    )

# Launch quietly
print("🚀 Starting Image Caption Generator...")
demo.launch(share=True, quiet=True, show_error=True)
