import { useState, useMemo, useRef, useCallback, Fragment } from "react";
import type { DialogResult, Mistake } from "@/types";
import { cn, LEVEL_COLORS, LEVEL_BG_COLORS, ERROR_TYPE_LABELS } from "@/lib/utils";

interface Props {
  result: DialogResult;
}

function parseSplittedAnswer(splitted: string): Map<number, string> {
  const sentences = new Map<number, string>();
  const regex = /\[(\d+)\]\s*/g;
  let match: RegExpExecArray | null;
  const positions: { index: number; num: number }[] = [];

  while ((match = regex.exec(splitted)) !== null) {
    positions.push({ index: match.index, num: parseInt(match[1]) });
  }

  for (let i = 0; i < positions.length; i++) {
    const start = positions[i].index + `[${positions[i].num}] `.length;
    const end = i + 1 < positions.length ? positions[i + 1].index : splitted.length;
    sentences.set(positions[i].num, splitted.slice(start, end));
  }

  return sentences;
}

function TextWithBreaks({ text }: { text: string }) {
  const parts = text.split("\n");
  return (
    <>
      {parts.map((line, i) => (
        <Fragment key={i}>
          {line}
          {i < parts.length - 1 && <br />}
        </Fragment>
      ))}
    </>
  );
}

export function AnswerViewer({ result }: Props) {
  const [hoveredMistake, setHoveredMistake] = useState<Mistake | null>(null);
  const [selectedMistake, setSelectedMistake] = useState<Mistake | null>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  const sentences = useMemo(
    () => parseSplittedAnswer(result.splitted_answer || ""),
    [result.splitted_answer]
  );

  const mistakesBySentence = useMemo(() => {
    const map = new Map<number, Mistake[]>();
    for (const m of result.mistakes) {
      for (const pos of m.position) {
        if (!map.has(pos)) map.set(pos, []);
        map.get(pos)!.push(m);
      }
    }
    return map;
  }, [result.mistakes]);

  const lastUserMessage = useMemo(() => {
    const dialog = result.dialog || [];
    for (let i = dialog.length - 1; i >= 0; i--) {
      if (dialog[i].role === "user") return dialog[i].content;
    }
    return dialog[dialog.length - 1]?.content || "";
  }, [result.dialog]);

  const scrollToMistake = useCallback((mistake: Mistake) => {
    if (!answerRef.current || mistake.position.length === 0) return;
    const sentenceNum = mistake.position[0];
    const el = answerRef.current.querySelector<HTMLElement>(`[data-sentence="${sentenceNum}"]`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, []);

  const handleMistakeClick = useCallback((mistake: Mistake) => {
    setSelectedMistake((prev) => {
      const next = prev === mistake ? null : mistake;
      if (next) scrollToMistake(next);
      return next;
    });
  }, [scrollToMistake]);

  const activeMistake = selectedMistake || hoveredMistake;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4 h-full">
      <div className="space-y-4 overflow-auto">
        <div className="bg-card border-2 border-primary/30 rounded-lg p-4 relative">
          <div className="text-xs font-bold uppercase text-muted-foreground mb-2">Промпт</div>
          <div className="text-sm whitespace-pre-wrap font-mono leading-relaxed max-h-[30vh] overflow-auto">
            {lastUserMessage}
          </div>
          <div className="absolute top-3 right-3 flex gap-2 text-xs">
            <span className="px-1.5 py-0.5 rounded bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-400 font-bold">
              L3: {result.mistakes_count["3"]}
            </span>
            <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-400 font-bold">
              L2: {result.mistakes_count["2"]}
            </span>
            <span className="px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-700 dark:bg-yellow-500/20 dark:text-yellow-400 font-bold">
              L1: {result.mistakes_count["1"]}
            </span>
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-xs font-bold uppercase text-muted-foreground mb-2">
            Ответ · {result.tokens} токенов
          </div>
          <div ref={answerRef} className="text-sm font-mono leading-relaxed">
            {Array.from(sentences.entries()).map(([num, text]) => {
              const mistakes = mistakesBySentence.get(num) || [];
              const maxLevel = mistakes.length > 0
                ? Math.max(...mistakes.map((m) => m.level))
                : 0;

              return (
                <span
                  key={num}
                  data-sentence={num}
                  className={cn(
                    "transition-colors",
                    mistakes.length > 0 && "cursor-pointer rounded px-0.5"
                  )}
                  style={
                    mistakes.length > 0
                      ? {
                          backgroundColor:
                            LEVEL_BG_COLORS[maxLevel as 1 | 2 | 3],
                          borderBottom: `2px solid ${LEVEL_COLORS[maxLevel as 1 | 2 | 3]}`,
                        }
                      : undefined
                  }
                  onMouseEnter={() => {
                    if (mistakes.length > 0) setHoveredMistake(mistakes[0]);
                  }}
                  onMouseLeave={() => setHoveredMistake(null)}
                  onClick={() => {
                    if (mistakes.length > 0) {
                      handleMistakeClick(mistakes[0]);
                    }
                  }}
                >
                  <TextWithBreaks text={text} />
                </span>
              );
            })}
          </div>
        </div>
      </div>

      <div className="bg-card border border-border rounded-lg p-4 overflow-auto">
        <div className="text-xs font-bold uppercase text-muted-foreground mb-3">
          Ошибки ({result.mistakes.length})
        </div>
        {result.mistakes.length === 0 ? (
          <div className="text-success font-bold text-center py-8">✓ Ошибок не найдено</div>
        ) : (
          <div className="space-y-2">
            {result.mistakes.map((m, i) => (
              <div
                key={i}
                className={cn(
                  "text-xs p-2 rounded cursor-pointer transition-all",
                  activeMistake === m && "ring-2 ring-primary"
                )}
                style={{ backgroundColor: LEVEL_BG_COLORS[m.level] }}
                onMouseEnter={() => setHoveredMistake(m)}
                onMouseLeave={() => setHoveredMistake(null)}
                onClick={() => handleMistakeClick(m)}
              >
                <div className="flex items-center gap-1 mb-1">
                  <span
                    className="px-1 py-0.5 rounded text-white font-bold text-[10px]"
                    style={{ backgroundColor: LEVEL_COLORS[m.level] }}
                  >
                    L{m.level}
                  </span>
                  <span className="font-medium text-foreground">
                    {ERROR_TYPE_LABELS[m.type] || m.type}
                  </span>
                  <span className="text-muted-foreground ml-auto">
                    [{m.position.join(", ")}]
                  </span>
                </div>
                <p className="text-muted-foreground leading-relaxed">{m.explanation}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
