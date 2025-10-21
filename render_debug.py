import argparse
import json
import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def load_benchmark_log(log_path: str) -> dict:
    """Загружает лог бенчмарка из JSON файла"""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_html(log_data: dict, filename: str) -> str:
    """Генерирует HTML для отладки используя Jinja2 шаблон"""
    
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('debug.html')
    
    config = log_data['config']
    results = log_data['results']
    summary = log_data['summary']
    
    # Подготавливаем данные для шаблона
    results_json = json.dumps(results, ensure_ascii=False)
    results_b64 = base64.b64encode(results_json.encode("utf-8")).decode("ascii")
    
    # Рендерим шаблон
    html = template.render(
        config=config,
        results=results,
        summary=summary,
        results_json=results_json,
        results_b64=results_b64,
        filename=filename
    )
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Генерация HTML для отладки логов бенчмарка')
    parser.add_argument('log_file', type=str, help='Путь к JSON логу бенчмарка')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Ошибка: файл {args.log_file} не найден")
        return
    
    print(f"Загрузка лога: {log_path}")
    log_data = load_benchmark_log(args.log_file)
    
    print("Генерация HTML...")
    html_content = generate_html(log_data, log_path.name)
    
    # Сохраняем HTML рядом с логом
    output_path = log_path.with_suffix('.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML сохранен в: {output_path}")
    print(f"\nОткройте файл в браузере для просмотра:")
    print(f"  {output_path.absolute()}")


if __name__ == "__main__":
    main()
