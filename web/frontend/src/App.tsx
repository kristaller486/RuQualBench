import { Routes, Route, Link } from "react-router-dom";
import { LeaderboardPage } from "@/pages/LeaderboardPage";
import { ViewerPage } from "@/pages/ViewerPage";
import { AdminPage } from "@/pages/AdminPage";
import { ThemeToggle } from "@/components/ThemeToggle";

export function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-border bg-card px-6 py-3 flex items-center justify-between">
        <Link to="/" className="text-lg font-bold tracking-tight">
          RuQualBench
        </Link>
        <ThemeToggle />
      </header>
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<LeaderboardPage />} />
          <Route path="/results/:modelId" element={<ViewerPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </main>
    </div>
  );
}
