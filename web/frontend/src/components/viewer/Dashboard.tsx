import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import type { ModelResults } from "@/types";
import {
  ERROR_TYPE_LABELS,
  ERROR_TYPE_COLORS,
  formatWithSE,
} from "@/lib/utils";

interface Props {
  data: ModelResults;
}

function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-card border border-border rounded-lg px-4 py-3">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-xl font-bold font-mono" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

export function Dashboard({ data }: Props) {
  const agg = data.aggregated;

  const errorTypeData = Object.entries(agg.error_types)
    .filter(([, v]) => v > 0)
    .map(([type, count]) => ({
      name: ERROR_TYPE_LABELS[type] || type,
      value: count,
      color: ERROR_TYPE_COLORS[type] || "#6b7280",
    }))
    .sort((a, b) => b.value - a.value);

  const barData = Object.entries(agg.error_types_by_level)
    .filter(([type]) => {
      const levels = agg.error_types_by_level[type];
      return levels["1"] + levels["2"] + levels["3"] > 0;
    })
    .map(([type]) => ({
      name: ERROR_TYPE_LABELS[type] || type,
      "L3": agg.error_types_by_level[type]["3"],
      "L2": agg.error_types_by_level[type]["2"],
      "L1": agg.error_types_by_level[type]["1"],
    }))
    .sort(
      (a, b) =>
        b["L3"] * 2 + b["L2"] + b["L1"] * 0.5 - (a["L3"] * 2 + a["L2"] + a["L1"] * 0.5)
    );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <MetricCard
          label="L3 (критичные) / 1K"
          value={formatWithSE(agg.l3_per_1k, agg.l3_per_1k_se)}
          color="#ef4444"
        />
        <MetricCard
          label="L2 (обычные) / 1K"
          value={formatWithSE(agg.l2_per_1k, agg.l2_per_1k_se)}
          color="#f59e0b"
        />
        <MetricCard
          label="L1 (незначит.) / 1K"
          value={formatWithSE(agg.l1_per_1k, agg.l1_per_1k_se)}
          color="#eab308"
        />
        <MetricCard
          label="Взвешенное / 1K"
          value={formatWithSE(agg.weighted_per_1k, agg.weighted_per_1k_se)}
        />
        <MetricCard
          label="Взвешенное / ответ"
          value={formatWithSE(agg.weighted_per_avg, agg.weighted_per_avg_se)}
        />
        <MetricCard
          label="Ср. длина ответа"
          value={`${agg.avg_tokens} tok`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card border border-border rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3">Распределение типов ошибок</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={errorTypeData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={100}
                innerRadius={50}
                paddingAngle={2}
                label={(props: any) => {
                  const { name, percent } = props;
                  return percent > 0.05 ? `${name} ${(percent * 100).toFixed(0)}%` : "";
                }}
                labelLine={false}
              >
                {errorTypeData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3">Ошибки по типам и уровням</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} layout="vertical">
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="L3" stackId="a" fill="#ef4444" />
              <Bar dataKey="L2" stackId="a" fill="#f59e0b" />
              <Bar dataKey="L1" stackId="a" fill="#eab308" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
