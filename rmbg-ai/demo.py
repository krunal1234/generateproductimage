import os
import uuid
import shutil
import torch
import requests

from PIL import Image
from huggingface_hub import hf_hub_download
from skimage import io
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image

app = FastAPI()

# Serve static files
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

def remove_img_bg_local(input_path: str, output_path: str):
    model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
    net = BriaRMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # prepare input
    model_input_size = [1024, 1024]
    orig_im = io.imread(input_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    # inference
    result = net(image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(input_path)
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image.save(output_path)
    return output_path

@app.get("/rmbg_from_url")
def remove_background_from_url(image_url: str = Query(..., description="URL of the image")):
    try:
        # Generate unique file name
        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"

        # Download image
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return JSONResponse(content={"error": "Failed to download image"}, status_code=400)

        with open(temp_input_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)

        # Remove background
        remove_img_bg_local(temp_input_path, temp_output_path)

        # Return result image URL
        output_url = f"/output/{os.path.basename(temp_output_path)}"
        return {"success": True, "url": output_url}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Old endpoint
@app.get("/rmbg")
def remove_background(name: str, type="jpg", outSuffix="_no_bg.png"):
    file_path = f"./resource/{name}.{type}"
    output_path = f"./resource/{name}{outSuffix}"
    remove_img_bg_local(file_path, output_path)
    return {"path": output_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
