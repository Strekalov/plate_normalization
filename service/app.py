import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.containers import AppContainer
from src.routes import plates as plates_routes
from src.routes.routers import router as app_router


def create_app() -> FastAPI:
    container = AppContainer()
    cfg = OmegaConf.load("config/config.yml")
    container.config.from_dict(cfg)
    container.wire([plates_routes])

    app = FastAPI()
    app.include_router(app_router, prefix="/plates", tags=["plates"])
    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, port=8888, host="0.0.0.0")
