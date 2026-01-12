"""
Роутеры API для RuQualBench Server.
"""

from server.routers.evaluate import router as evaluate_router
from server.routers.jobs import router as jobs_router

__all__ = ["evaluate_router", "jobs_router"]
