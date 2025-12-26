import json
import re
import logging
import nltk
import tiktoken
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_dataset(dataset_name: str) -> List[List[Dict[str, str]]]:
    """Загружает датасет из json файла"""
    sizes = {'lite': 100, 'base': 250, 'large': 500, "debug": 5}
    filename = f"{dataset_name}_bench_{sizes[dataset_name]}.json"
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_tokens(text: str) -> int:
    """Подсчитывает токены используя o200k_base encoding"""
    if not text:
        return 0
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def extract_json_from_response(text: str) -> Any:
    """Извлекает JSON из ответа, который может быть обернут в markdown code block"""
    # Сначала пробуем найти JSON блок в markdown
    json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Если не нашли или не распарсили, пробуем найти объект {...} или список [...]
    # Ищем первое вхождение { или [
    start_match = re.search(r'[\[\{]', text)
    if start_match:
        start_idx = start_match.start()
        # Пытаемся распарсить начиная с найденной скобки
        try:
            return json.loads(text[start_idx:])
        except json.JSONDecodeError:
            # Если не вышло, пробуем найти закрывающую скобку и обрезать
            # Это грубый метод, но может сработать для простых случаев
            pass
            
    # Пробуем найти просто {...}
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Пробуем найти просто [...]
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Последняя попытка - распарсить весь текст
    return json.loads(text)

def remove_think_tags(text: str) -> str:
    """Удаляет контент между тегами <think></think>"""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def ensure_nltk_resources():
    """Проверяет и скачивает необходимые ресурсы NLTK"""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Скачивание ресурса nltk 'punkt_tab'...")
        nltk.download('punkt_tab')

def split_into_numbered_sentences(text: str) -> str:
    """Разбивает текст на предложения и нумерует их, сохраняя форматирование"""
    if not text:
        return ""
        
    ensure_nltk_resources()
    
    # Получаем спаны (начало, конец) для каждого предложения
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
        spans = list(tokenizer.span_tokenize(text))
    except LookupError:
        # Fallback if russian pickle not found, try downloading punkt
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
        spans = list(tokenizer.span_tokenize(text))
    
    result = []
    prev_end = 0
    
    for i, (start, end) in enumerate(spans, 1):
        # Добавляем текст между предложениями (пробелы, переносы строк) как есть
        result.append(text[prev_end:start])
        
        # Добавляем номер перед предложением
        result.append(f"[{i}] ")
        
        # Добавляем само предложение
        result.append(text[start:end])
        
        prev_end = end
        
    # Добавляем оставшийся хвост текста (если есть)
    result.append(text[prev_end:])
    
    return "".join(result)