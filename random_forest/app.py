from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import cv2
import numpy as np
import os
import pandas as pd
from scipy.stats import mode

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = load('jaundice-detector.joblib')

# Calculate the mode values
def safe_mode(values):
    mode_result = mode(values)
    if mode_result.mode > 0:
        return mode_result.mode
    else:
        return None


def extract_rgb_ycc(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None, None, None, None, None, None
    
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Initialize lists to store the values
    red_values = []
    green_values = []
    blue_values = []
    y_values = []
    cr_values = []
    cb_values = []

    for row in range(height):
        for col in range(width):
            pixel = image[row, col]
            blue, green, red = pixel
            red_values.append(red)
            green_values.append(green)
            blue_values.append(blue)
            # Convert RGB to YCrCb
            ycrcb_pixel = cv2.cvtColor(np.array([[pixel]], dtype=np.uint8), cv2.COLOR_BGR2YCrCb)
            y, cr, cb = ycrcb_pixel[0][0]
            y_values.append(y)
            cr_values.append(cr)
            cb_values.append(cb)
    
    mode_red = safe_mode(red_values)
    mode_green = safe_mode(green_values)
    mode_blue = safe_mode(blue_values)
    mode_y = safe_mode(y_values)
    mode_cr = safe_mode(cr_values)
    mode_cb = safe_mode(cb_values)

    return mode_red, mode_green, mode_blue, mode_y, mode_cr, mode_cb


@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    image_path = f"/tmp/{image.filename}"
    
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    # Extract RGB and YCrCb values
    red_values, green_values, blue_values, y_values, cr_values, cb_values = extract_rgb_ycc(image_path)
    print(red_values)
    print(green_values)

    if red_values is None:
        raise HTTPException(status_code=500, detail="Failed to process image")

    df = pd.DataFrame({'red':[red_values], 'green':[red_values], 'blue':[blue_values], 'Y':[y_values],  'cblue':[cb_values],  'cred':[cr_values]})
    prediction = model.predict(df)
    print(prediction)

    # Determine the result based on the prediction
    if prediction == 1:
        result = 'Not Jaundiced'
    elif prediction == 2:
        result = 'Jaundiced'
    else:
        result = 'Unknown result'

    return {"prediction": int(prediction[0]), "result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
