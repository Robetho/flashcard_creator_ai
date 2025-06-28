# api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os

# Import your flashcard generation logic
# Assuming flashcard_generator.py is in the same directory
try:
    from flashcard_generator import generate_mcq_flashcards
    # You might need to adjust the path or structure if flashcard_generator.py is moved
except ImportError:
    print("Error: flashcard_generator.py not found or has errors.")
    print("Please ensure 'flashcard_generator.py' is in the same directory.")
    sys.exit(1) # Exit if the core logic can't be imported

app = FastAPI(
    title="EduAI Flashcard API",
    description="API for generating AI-powered flashcards from text."
)

class TextInput(BaseModel):
    text: str

@app.post("/generate_flashcards/")
async def create_flashcards(data: TextInput):
    """
    Generates MCQ flashcards from the provided text.
    """
    if not data.text or len(data.text) < 50: # Basic validation
        raise HTTPException(status_code=400, detail="Input text is too short or empty.")

    try:
        flashcards = generate_mcq_flashcards(data.text)
        if not flashcards:
            # If no flashcards were generated but text was valid
            return {"message": "No flashcards could be generated from the provided text. Try a different text.", "flashcards": []}
        return {"flashcards": flashcards}
    except Exception as e:
        # Catch any errors during flashcard generation
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# To run this file: uvicorn api_server:app --reload
# --reload enables auto-reloading on code changes (useful for development)
# --host 0.0.0.0 makes it accessible from other devices on your local network
# if you are testing from a physical phone. For emulator, localhost is fine.

if __name__ == "__main__":
    # This block is for running the API server directly from this file.
    # For production, you'd use a process manager like Gunicorn or PM2.
    # For local development, running it via uvicorn command is better.
    print("To run the API server, use the command:")
    print("uvicorn api_server:app --reload --host 0.0.0.0 --port 8000")
    print("Make sure your virtual environment is activated.")