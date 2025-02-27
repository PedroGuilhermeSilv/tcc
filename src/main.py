from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from converter.routes import video_routes, model_routes
from database import create_tables

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_routes.router)
app.include_router(model_routes.router)


@app.get("/init-db")
async def initialize_database():
    create_tables()
    return {"message": "Database initialized successfully"}


# Add this to make the app discoverable
if __name__ == "__main__":
    import uvicorn

    # Tentar criar as tabelas na inicialização
    try:
        create_tables()
        print("Tabelas criadas com sucesso!")
    except Exception as e:
        print(f"Erro ao criar tabelas: {e}")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
