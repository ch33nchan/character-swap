#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from PIL import Image
from loguru import logger

# Import upload function from lighting_transfer
from tools.lighting_transfer import upload_image_to_storage

def upload_and_get_url(image, prefix="single"):
    """Upload image to Azure and return the URL"""
    if image is None:
        return "No image uploaded", None
    
    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        
        # Upload to Azure
        logger.info(f"Uploading image to Azure with prefix: {prefix}")
        url = upload_image_to_storage(image, prefix=prefix)
        
        logger.success(f"Upload successful: {url}")
        return url, image
        
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        logger.error(error_msg)
        return error_msg, None

def upload_batch(files, prefix="batch"):
    """Upload multiple images and return URLs"""
    if not files:
        return "No files uploaded"
    
    results = []
    for idx, file in enumerate(files):
        try:
            img = Image.open(file).convert("RGB")
            url = upload_image_to_storage(img, prefix=prefix)
            results.append(f"{idx+1}. {url}")
            logger.success(f"Uploaded {idx+1}/{len(files)}: {url}")
        except Exception as e:
            results.append(f"{idx+1}. ERROR: {str(e)}")
            logger.error(f"Failed to upload file {idx+1}: {e}")
    
    return "\n".join(results)

# Create Gradio interface
with gr.Blocks(title="Azure Image Uploader") as app:
    gr.Markdown("# 🔗 Azure Blob Storage Image Uploader")
    gr.Markdown("Upload images and get Azure CDN URLs instantly")
    
    with gr.Tabs():
        with gr.TabItem("Single Upload"):
            with gr.Row():
                with gr.Column():
                    single_image = gr.Image(type="pil", label="Upload Image")
                    single_prefix = gr.Textbox(
                        value="single",
                        label="Storage Prefix",
                        placeholder="e.g., single, test, demo"
                    )
                    single_btn = gr.Button("Upload & Get URL", variant="primary")
                
                with gr.Column():
                    single_url = gr.Textbox(
                        label="Azure CDN URL",
                        placeholder="URL will appear here...",
                        lines=3
                    )
                    single_preview = gr.Image(label="Preview", type="pil")
            
            single_btn.click(
                upload_and_get_url,
                inputs=[single_image, single_prefix],
                outputs=[single_url, single_preview]
            )
        
        with gr.TabItem("Batch Upload"):
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Multiple Images"
                    )
                    batch_prefix = gr.Textbox(
                        value="batch",
                        label="Storage Prefix",
                        placeholder="e.g., batch, collection, dataset"
                    )
                    batch_btn = gr.Button("Upload All & Get URLs", variant="primary")
                
                with gr.Column():
                    batch_urls = gr.Textbox(
                        label="Azure CDN URLs",
                        placeholder="URLs will appear here...",
                        lines=15
                    )
            
            batch_btn.click(
                upload_batch,
                inputs=[batch_files, batch_prefix],
                outputs=batch_urls
            )
    
    gr.Markdown("""
    ### 📝 Notes:
    - Images are uploaded to Azure Blob Storage
    - URLs are served via CDN for fast access
    - Supported formats: PNG, JPG, JPEG, WebP
    - Storage prefix helps organize your files
    """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7878)
