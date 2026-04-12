from fastapi import FastAPI
from fastapi.responses import JSONResponse
# from pydantic import BaseModel
import uvicorn

from python.preprocess import input_preprocess
from python.response import LLM_response

app = FastAPI()

@app.get("/")
async def health_check():
    return JSONResponse(content={"status":"API Running"} ,status_code=200 )


@app.post("/report")
async def generate_report():
    inputData = await input_preprocess()
    res = await LLM_response(inputData)

    return res

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5000, reload=True)