from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from diffusers import StableDiffusionPipeline
import numpy as np

app = FastAPI()

# Output directory
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
net = None
stable_diffusion_pipe = None

# Startup: Load models
@app.on_event("startup")
def load_model():
    global net, stable_diffusion_pipe

    # Load background removal model
    model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
    net = BriaRMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # Load Stable Diffusion
    stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    stable_diffusion_pipe.to(device)

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Image processing endpoint
@app.post("/generate_product_image")
async def generate_product_image(
    image_id: str = Form(...),
    background_prompt: str = Form(...)
):
    try:
        temp_output_path = f"{OUTPUT_DIR}/{image_id}_no_bg.png"
        final_output_path = f"{OUTPUT_DIR}/{image_id}_final_output.png"

        if not os.path.exists(temp_output_path):
            return JSONResponse(content={"error": "No background-removed image found for given ID"}, status_code=404)

        product_image = Image.open(temp_output_path).convert("RGBA")
        background_image = generate_background_with_stable_diffusion(background_prompt)
        final_image = combine_images(background_image, product_image)

        final_image.save(final_output_path)

        output_url = f"/output/{os.path.basename(final_output_path)}"

        return {
            "success": True,
            "url": output_url
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/remove_background")
async def remove_background(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)

        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"

        with open(temp_input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        remove_img_bg_local(temp_input_path, temp_output_path)

        no_bg_url = f"/output/{os.path.basename(temp_output_path)}"

        return {
            "success": True,
            "no_bg_url": no_bg_url,
            "image_id": temp_id  # Optional, used for next step
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Helper functions
def remove_img_bg_local(input_path: str, output_path: str):
    if net is None:
        raise ValueError("Background removal model not loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    with Image.open(input_path) as pil_image:
        pil_image = pil_image.convert("RGB")
        orig_im = np.array(pil_image)

    model_input_size = [1024, 1024]
    image = preprocess_image(orig_im, model_input_size).to(device)
    result = net(image)

    result_image = postprocess_image(result[0][0], orig_im.shape[0:2])
    mask = Image.fromarray(result_image).convert("L")
    orig_image = Image.open(input_path).convert("RGBA")

    no_bg_image = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_image, mask=mask)
    no_bg_image.save(output_path)

    return output_path


def generate_background_with_stable_diffusion(prompt: str) -> Image:
    generator = torch.manual_seed(42)
    image = stable_diffusion_pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images[0]
    return image


def combine_images(background_image: Image, product_image: Image) -> Image:
    product_image = product_image.resize(background_image.size, Image.ANTIALIAS)
    background_image.paste(product_image, (0, 0), product_image)
    return background_image


# Mount output directory
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
