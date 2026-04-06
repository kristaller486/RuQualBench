import os
from pathlib import Path
from ipaddress import ip_address, ip_network
from typing import List, Union
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Пароль для админки (обязателен)
    admin_password: str = ""
    # Белый список IP для админки (CIDR или одиночные адреса)
    ip_whitelist: List[Union[str]] = field(default_factory=list)
    # Путь к директории с V2-логами
    logs_v2_dir: Path = Path("logs_v2")
    # URL PostgreSQL базы
    database_url: str = "postgresql+psycopg://ruqualbench:ruqualbench@localhost:5432/ruqualbench"
    # Путь к корню git-репозитория (для git pull)
    repo_path: Path = Path(".")
    # Порт
    port: int = 8080

    def check_ip(self, client_ip: str) -> bool:
        """Проверяет, входит ли IP в белый список."""
        if not self.ip_whitelist:
            return True
        try:
            addr = ip_address(client_ip)
        except ValueError:
            return False
        for entry in self.ip_whitelist:
            try:
                if "/" in entry:
                    if addr in ip_network(entry, strict=False):
                        return True
                else:
                    if addr == ip_address(entry):
                        return True
            except ValueError:
                continue
        return False


def load_settings() -> Settings:
    """Загружает настройки из переменных окружения."""
    whitelist_raw = os.getenv("IP_WHITELIST", "")
    whitelist = [s.strip() for s in whitelist_raw.split(",") if s.strip()]

    logs_dir = os.getenv("LOGS_V2_DIR", "logs_v2")
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://ruqualbench:ruqualbench@localhost:5432/ruqualbench",
    )
    repo_path = os.getenv("REPO_PATH", ".")
    port = int(os.getenv("PORT", "8080"))

    return Settings(
        admin_password=os.getenv("ADMIN_PASSWORD", ""),
        ip_whitelist=whitelist,
        logs_v2_dir=Path(logs_dir),
        database_url=database_url,
        repo_path=Path(repo_path),
        port=port,
    )
