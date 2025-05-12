from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow
from model import predict_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_upload_form():
    return HTMLResponse("""
        <html>
            <body>
                <form action="/upload-image/" method="post" enctype="multipart/form-data">
                    <input type="file" name="image" />
                    <input type="submit" value="Upload Image" />
                </form>
            </body>
        </html>
    """)

@app.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    # Read the image file
    image_data = await image.read()

    # Open the image with Pillow
    image_pil = Image.open(BytesIO(image_data))

    # Process the image (you can add your image processing code here)
    image_pil = image_pil.convert("RGB")

    # Optionally save or manipulate the image
    image_pil.save("./uploads/tumor.jpg")
    
    val = predict_image("./uploads/tumor.jpg")
    
    

    return {"filename": image.filename, "tumor type" : val}
