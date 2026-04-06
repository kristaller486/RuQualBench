import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SyncService:
    """Сервис для синхронизации логов через git pull."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.last_sync: Optional[datetime] = None
        self.last_sync_output: str = ""
        self.is_syncing: bool = False

    async def pull(self) -> str:
        """Выполняет git pull и возвращает вывод."""
        if self.is_syncing:
            return "Синхронизация уже выполняется"

        self.is_syncing = True
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "pull",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n" + stderr.decode("utf-8", errors="replace")

            self.last_sync = datetime.now()
            self.last_sync_output = output.strip()
            logger.info(f"Git pull выполнен: {self.last_sync_output}")
            return self.last_sync_output
        except Exception as e:
            msg = f"Ошибка git pull: {e}"
            logger.error(msg)
            self.last_sync_output = msg
            return msg
        finally:
            self.is_syncing = False

    def get_status(self) -> dict:
        return {
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "last_output": self.last_sync_output,
            "is_syncing": self.is_syncing,
        }
