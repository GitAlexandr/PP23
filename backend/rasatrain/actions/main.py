from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from minio.error import S3Error

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
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

rasa_model_path = "/home/sirius/Рабочий стол/rasatrain/models"
rasa_agent = Agent.load(rasa_model_path)

# MinIO configuration
minio_client = Minio(
    "minio_server_address",
    access_key="your_access_key",
    secret_key="your_secret_key",
    secure=False
)
minio_bucket_name = "your_bucket_name"


@app.post("/chat")
async def chat(message: Message):
    try:
        response = await rasa_agent.handle_text(message.message)
        return {"response": response[0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        minio_client.fput_object(minio_bucket_name, file.filename, file.file, length=file.file._file.fstat().st_size)
        return {"filename": file.filename}
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
