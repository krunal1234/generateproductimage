import logging
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import os
import shutil
import uuid
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image

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
        net = BriaRMBG().to(device)
        logging.info("Bria RMBG model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Bria RMBG model: {e}")

    # Load Stable Diffusion model
    try:
        logging.info("Loading Stable Diffusion pipeline...")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=dtype
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


# Route for background removal
@app.post("/remove_background")
async def remove_background(file: UploadFile = File(...)):
    try:
        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"

        # Validate uploaded file content type
        if not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)

        # Save uploaded image to disk
        with open(temp_input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Remove background from the uploaded image
        remove_img_bg(temp_input_path, temp_output_path)

        output_url = f"/output/{os.path.basename(temp_output_path)}"
        return {"success": True, "url": output_url}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Route for generating background with Stable Diffusion
@app.post("/generate_background")
async def generate_background(background_prompt: str = Query(...)):
    try:
        # Generate background using Stable Diffusion
        background_image = generate_background_with_stable_diffusion(background_prompt)

        # Save the generated background image
        temp_id = str(uuid.uuid4())
        final_output_path = f"{OUTPUT_DIR}/{temp_id}_background.png"
        background_image.save(final_output_path)

        output_url = f"/output/{os.path.basename(final_output_path)}"
        return {"success": True, "url": output_url}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Background removal function
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


# Stable Diffusion background generation
def generate_background_with_stable_diffusion(prompt: str) -> Image:
    if stable_diffusion_pipe is None:
        raise ValueError("Stable Diffusion pipeline not loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <-- Add this
    generator = torch.Generator(device=device).manual_seed(42)
    
    image = stable_diffusion_pipe(
        prompt,
        guidance_scale=7.5,
        num_inference_steps=50,
        generator=generator
    ).images[0]

    return image

# Mount output folder
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
