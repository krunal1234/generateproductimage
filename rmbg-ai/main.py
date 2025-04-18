from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import os
import shutil
import uuid
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from utilities import preprocess_image, postprocess_image
from diffusers import StableDiffusionPipeline
import numpy as np
from briarmbg import BriaRMBG

app = FastAPI()

# Output directory setup
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for models
net = None
stable_diffusion_pipe = None

# Pre-load models on startup

@app.on_event("startup")
def load_model():
    global net, stable_diffusion_pipe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Bria RMBG model
    try:
        logging.info("Loading Bria RMBG model...")
        model_path = hf_hub_download(repo_id="briaai/RMBG-1.4", filename='model.pth')
        net = BriaRMBG()
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        logging.info("Bria RMBG model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Bria RMBG model: {e}")

    # Load Stable Diffusion model
    try:
        logging.info("Loading Stable Diffusion pipeline...")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=dtype
        )
        stable_diffusion_pipe.to(device)
        logging.info("Stable Diffusion pipeline loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Stable Diffusion pipeline: {e}")
        
# Route to serve index.html
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Route for product image background removal and background generation
@app.post("/product_image_display")
async def product_image_display(file: UploadFile = File(...), background_prompt: str = Query(...)):
    try:
        # Create unique identifier for file handling
        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"
        final_output_path = f"{OUTPUT_DIR}/{temp_id}_final_output.png"

        # Validate uploaded file content type
        if not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)

        # Save uploaded image to disk
        with open(temp_input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Remove background from the uploaded product image
        remove_img_bg(temp_input_path, temp_output_path)

        # Generate a suitable background using Stable Diffusion
        product_image = Image.open(temp_output_path).convert("RGBA")

        # Generate background using Stable Diffusion based on the given prompt
        background_image = generate_background_with_stable_diffusion(background_prompt)

        # Combine the background with the product image
        final_image = combine_images(background_image, product_image)

        # Save final image
        final_image.save(final_output_path)

        output_url = f"/output/{os.path.basename(final_output_path)}"
        return {"success": True, "url": output_url}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def remove_img_bg(input_path: str, output_path: str):
    if net is None:
        raise ValueError("Background removal model not loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Read image
    with Image.open(input_path) as pil_image:
        pil_image = pil_image.convert("RGB")
        orig_im = np.array(pil_image)

    if orig_im is None or orig_im.size == 0:
        raise ValueError("Failed to load image for processing")

    # Prepare input and move to the correct device
    model_input_size = [1024, 1024]
    image = preprocess_image(orig_im, model_input_size).to(device)
    result = net(image)

    # Post-process and remove background
    result_image = postprocess_image(result[0][0], orig_im.shape[0:2])
    mask = Image.fromarray(result_image).convert("L")
    orig_image = Image.open(input_path).convert("RGBA")

    no_bg_image = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_image, mask=mask)
    no_bg_image.save(output_path)

def generate_background_with_stable_diffusion(prompt: str) -> Image:
    # Generate background with Stable Diffusion
    generator = torch.manual_seed(42)  # Set seed for reproducibility
    image = stable_diffusion_pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images[0]
    return image

def combine_images(background_image: Image, product_image: Image) -> Image:
    # Resize product image to fit the background size (Optional)
    product_image = product_image.resize(background_image.size, Image.ANTIALIAS)

    # Composite the images (place the product image over the background)
    background_image.paste(product_image, (0, 0), product_image)
    return background_image

# Mount output folder
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)