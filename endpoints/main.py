from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from rasa.core.agent import Agent
import uvicorn

app = FastAPI()

class Message(BaseModel):
    message: str


rasa_model_path = "/Users/sasha/Desktop/pp/rasaallll/models"
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