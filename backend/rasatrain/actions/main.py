from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from rasa.core.agent import Agent
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse



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


@app.post("/chat")
async def chat(message: Message):
    try:
        response = await rasa_agent.handle_text(message.message)
        return {"response": response[0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 