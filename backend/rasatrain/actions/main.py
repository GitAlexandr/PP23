from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel
from rasa.core.agent import Agent
import tempfile
import os
import shutil

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:9000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

rasa_model_path = "../models"
rasa_agent = Agent.load(rasa_model_path)

# MinIO configuration
minio_client = Minio(
    "127.0.0.1:9000",
    access_key="minioaccesskey",
    secret_key="miniosecretkey",
    secure=False
)
minio_bucket_name = "backet"


@app.post("/chat")
async def chat(message: Message):
    try:
        response = await rasa_agent.handle_text(message.message)
        return {"response": response[0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            
            if temp_file_path:
                file_size = os.stat(temp_file_path).st_size
                minio_client.fput_object(minio_bucket_name, file.filename, temp_file_path)
                
                file_url = minio_client.presigned_get_object(minio_bucket_name, file.filename)
                
                return {"filename": file.filename, "file_url": file_url}
            else:
                raise HTTPException(status_code=500, detail="Failed to upload file: Invalid temporary file path")
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    finally:
        if temp_file_path:
            os.unlink(temp_file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
