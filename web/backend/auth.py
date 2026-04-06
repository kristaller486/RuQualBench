from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from web.backend.config import Settings


class AdminAuthMiddleware(BaseHTTPMiddleware):
    """Проверяет пароль и IP для admin-эндпоинтов."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/admin"):
            # Проверяем IP
            client_ip = request.client.host if request.client else ""
            if not self.settings.check_ip(client_ip):
                raise HTTPException(status_code=403, detail="IP не в белом списке")

            # Проверяем пароль
            if self.settings.admin_password:
                token = request.headers.get("X-Admin-Token", "")
                if token != self.settings.admin_password:
                    raise HTTPException(status_code=401, detail="Неверный пароль")

        return await call_next(request)
