import { Sun, Moon, Monitor } from "lucide-react";
import { useTheme } from "@/lib/useTheme";
import { cn } from "@/lib/utils";

const options = [
  { value: "light" as const, icon: Sun, title: "Светлая" },
  { value: "dark" as const, icon: Moon, title: "Тёмная" },
  { value: "system" as const, icon: Monitor, title: "Системная" },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="flex items-center bg-secondary rounded-lg p-0.5 gap-0.5">
      {options.map(({ value, icon: Icon, title }) => (
        <button
          key={value}
          onClick={() => setTheme(value)}
          title={title}
          className={cn(
            "p-1.5 rounded-md transition-colors",
            theme === value
              ? "bg-card text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          <Icon className="w-3.5 h-3.5" />
        </button>
      ))}
    </div>
  );
}
