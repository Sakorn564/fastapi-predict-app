from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from nyoka import PMMLModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load PMML model
model = PMMLModel.load(open("model.pmml", "rb"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(contents)
    predictions = model.predict(df)
    df['prediction'] = predictions
    return JSONResponse(content=df.to_dict(orient="records"))
