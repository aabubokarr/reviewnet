"use client";

import { FileText, Download } from "lucide-react";
import { Button } from "@/components/ui/button";

const reports = [
  { id: 1, name: "Full Analysis Report - Q4 2025", type: "Full Report", date: "2026-01-15", status: "Ready" },
  { id: 2, name: "Sentiment Analysis - TikTok", type: "Sentiment", date: "2026-02-10", status: "Ready" },
  { id: 3, name: "Toxicity Report - Twitter/X", type: "Toxicity", date: "2026-02-18", status: "Ready" },
  { id: 4, name: "Emotion Analysis - All Apps", type: "Emotion", date: "2026-02-19", status: "Generating" },
];

export default function ReportsPage() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Reports</h1>
          <p className="mt-1 text-muted-foreground">Generate and export analysis reports.</p>
        </div>
        <Button className="gradient-primary border-0">
          <FileText className="mr-2 h-4 w-4" /> Generate Report
        </Button>
      </div>

      <div className="glass-card rounded-xl p-5">
        <h3 className="mb-4 text-lg font-semibold">Report History</h3>
        <div className="space-y-3">
          {reports.map((r) => (
            <div key={r.id} className="flex items-center justify-between rounded-lg border border-border/50 p-4 hover:bg-accent/30 transition-colors">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <FileText className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="font-medium text-sm">{r.name}</p>
                  <p className="text-xs text-muted-foreground">{r.type} · {r.date}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-medium ${r.status === "Ready" ? "text-success" : "text-warning animate-pulse-soft"}`}>
                  {r.status}
                </span>
                {r.status === "Ready" && (
                  <Button variant="ghost" size="sm" className="h-8">
                    <Download className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
