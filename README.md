# RuQualBench 🐸

## Описание

Бенчмарк для оценки качества русского языка у LLM. Через LLM-судью считаем количество типичных для LLM ошибок в ответах на набор случайных вопросов. Вопсросы были взяты из этих датасетов:

 - [kristaller486/wikisource_preferences_ru](https://huggingface.co/datasets/kristaller486/wikisource_preferences_ru) (gpt-4.1-mini-orig-segment-score > 4)
 - [Arketov/kalo_misc_part2_no_system_ru](https://huggingface.co/datasets/Arketov/kalo_misc_part2_no_system_ru)
 - [kristaller486/writingprompts-ru](https://huggingface.co/datasets/kristaller486/writingprompts-ru)
 - [t-tech/T-Wix](https://huggingface.co/datasets/t-tech/T-Wix) (subset == general)

В качестве судьи рекомендуется использовать Gemini 2.5 Pro, рекомендуется делать не менее трех запусков (`-n 3`) бенчмарка из-за разброса между оценками судьи.

Промт был оптимизирован через ответы Gemini 2.5 Flash Lite (GA), ответы этой модели могут быть слегка завышены.

## Как использовать

### Запуск бенчмарка

```bash
uv run python main.py --help
```

``` 
usage: main.py v1 [-h] [--dataset {debug,lite,base,large}] [--model MODEL] [--judge-model JUDGE_MODEL] [--extra-body EXTRA_BODY] [-n NUM_RUNS] [-v VERBOSE_NAME]
                  [--continue CONTINUE_TIMESTAMP] [--no-regenerate]

options:
  -h, --help            show this help message and exit
  --dataset {debug,lite,base,large}
                        Выбор датасета (по умолчанию: lite)
  --model MODEL         Переопределить тестируемую модель из .env
  --judge-model JUDGE_MODEL
                        Переопределить модель-оценщик из .env
  --extra-body EXTRA_BODY
                        JSON объект для extra_body параметра тестируемой модели (например: '{"temperature": 0.7}')
  -n, --num-runs NUM_RUNS
                        Количество прогонов бенчмарка для вычисления средних значений и погрешности (по умолчанию: 1)
  -v, --verbose-name VERBOSE_NAME
                        Красивое имя модели для отображения в лидерборде (опционально)
  --continue CONTINUE_TIMESTAMP
                        Продолжить существующую серию прогонов (указать timestamp, например: 2025-10-17_15-17-05)
  --no-regenerate       Генерировать ответы от модели только один раз, оценивать судьей N раз (работает с -n)
```

### Посмотреть результаты

```bash
uv run python render_debug.py --help
```

```
usage: render_debug.py [-h] log_file

Генерация HTML для отладки логов бенчмарка

positional arguments:
  log_file    Путь к JSON логу бенчмарка

options:
  -h, --help  show this help message and exit
```

### Сборка лидерборда

```bash
uv run python generate_leaderboard.py
```

### Режим сервеора / API для бенчмарка
```bash
uv run python -m server
```

## Citing RuQualBench

```
@misc{kristaller
   author = {kristaller},
   title = {RuQualBench: A benchmark for evaluating the quality of the Russian language in LLM responses}
   url = {https://github.com/kristaller486/ruqualbench}
}
```