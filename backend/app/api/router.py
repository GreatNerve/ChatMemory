from fastapi import APIRouter

from app.api.routes import ask, jobs, people, persona, settings, workspaces

api_router = APIRouter()
api_router.include_router(settings.router)
api_router.include_router(workspaces.router)
api_router.include_router(people.router)
api_router.include_router(ask.router)
api_router.include_router(persona.router)
api_router.include_router(jobs.router)
