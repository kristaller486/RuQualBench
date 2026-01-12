"""
Точка входа для запуска сервера как модуля.

Использование:
    python -m server
    python -m server --host 0.0.0.0 --port 8000
    python -m server --reload
    python -m server --workers 4 --job-retention-hours 48
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="RuQualBench Server - API для оценки качества русского языка в ответах LLM"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для запуска сервера (по умолчанию: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для запуска сервера (по умолчанию: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Включить автоперезагрузку при изменении файлов (для разработки)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Количество воркеров uvicorn (по умолчанию: 1, игнорируется при --reload)"
    )
    parser.add_argument(
        "--job-retention-hours",
        type=int,
        default=24,
        help="Время хранения завершённых асинхронных задач в часах (по умолчанию: 24)"
    )
    
    args = parser.parse_args()
    
    # Передаём настройки через переменные окружения для использования в приложении
    import os
    os.environ["_SERVER_JOB_RETENTION_HOURS"] = str(args.job_retention_hours)
    
    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
