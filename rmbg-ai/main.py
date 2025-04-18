from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import torch
from PIL import Image
from briarmbg import BriaRMBG
from huggingface_hub import hf_hub_download
from utilities import preprocess_image, postprocess_image
from diffusers import AutoPipelineForText2Image
import numpy as np

app = FastAPI()

# Setup CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model variables
net = None
stable_diffusion_pipe = None

@app.on_event("startup")
def load_model():
    global net, stable_diffusion_pipe

    try:
        print("🔄 Downloading BriaRMBG model...")
        model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')

        net = BriaRMBG()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        print("✅ BriaRMBG model loaded.")
    except Exception as e:
        print(f"❌ Failed to load BriaRMBG model: {e}")
        raise

    try:
        print("🔄 Loading Stable Diffusion pipeline from Hugging Face...")
        stable_diffusion_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
stable_diffusion_pipe.to(device
        print("✅ Stable Diffusion pipeline ready.")
    except Exception as e:
        print(f"❌ Failed to load Stable Diffusion pipeline: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/product_image_display")
async def product_image_display(file: UploadFile = File(...), background_prompt: str = Query(...)):
    try:
        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"
        final_output_path = f"{OUTPUT_DIR}/{temp_id}_final_output.png"

        if not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)

        # Save uploaded file
        with open(temp_input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Remove background
        remove_img_bg_local(temp_input_path, temp_output_path)

        # Load processed image
        product_image = Image.open(temp_output_path).convert("RGBA")

        # Generate background
        background_image = generate_background_with_stable_diffusion(background_prompt)

        # Combine product with background
        final_image = combine_images(background_image, product_image)
        final_image.save(final_output_path)

        output_url = f"/output/{os.path.basename(final_output_path)}"
        return {"success": True, "url": output_url}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def remove_img_bg_local(input_path: str, output_path: str):
    if net is None:
        raise ValueError("Background removal model not loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    with Image.open(input_path) as pil_image:
        pil_image = pil_image.convert("RGB")
        orig_im = np.array(pil_image)

    if orig_im is None or orig_im.size == 0:
        raise ValueError("Invalid image")

    model_input_size = [1024, 1024]
    image = preprocess_image(orig_im, model_input_size).to(device)

    with torch.no_grad():
        result = net(image)

    result_image = postprocess_image(result[0][0], orig_im.shape[0:2])
    mask = Image.fromarray(result_image).convert("L")
    orig_image = Image.open(input_path).convert("RGBA")

    no_bg_image = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_image, mask=mask)
    no_bg_image.save(output_path)

def generate_background_with_stable_diffusion(prompt: str) -> Image:
    if stable_diffusion_pipe is None:
        raise ValueError("Stable Diffusion pipeline not initialized")

    try:
        with torch.no_grad():
            generator = torch.manual_seed(42)
            result = stable_diffusion_pipe(prompt, guidance_scale=7.5, num_inference_steps=50)
            return result.images[0]
    except Exception as e:
        raise RuntimeError(f"Stable Diffusion generation failed: {str(e)}")

def combine_images(background_image: Image, product_image: Image) -> Image:
    product_image = product_image.resize(background_image.size, Image.ANTIALIAS)
    background_image.paste(product_image, (0, 0), product_image)
    return background_image

# Serve static files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Local run entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)