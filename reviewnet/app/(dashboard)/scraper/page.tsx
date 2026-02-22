"use client";

import { useState } from "react";
import { Download, Play, Loader2, CheckCircle2, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const scrapingHistory = [
  { id: 1, appId: "com.tiktok.android", status: "completed", reviews: 4280, date: "2026-02-18" },
  { id: 2, appId: "com.instagram.android", status: "completed", reviews: 3950, date: "2026-02-17" },
  { id: 3, appId: "com.whatsapp", status: "completed", reviews: 3420, date: "2026-02-16" },
  { id: 4, appId: "com.spotify.music", status: "in_progress", reviews: 1240, date: "2026-02-20" },
];

export default function ScraperPage() {
  const [appId, setAppId] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  const handleStart = () => {
    if (!appId) return;
    setIsRunning(true);
    setTimeout(() => setIsRunning(false), 3000);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold">Data Scraper</h1>
        <p className="mt-1 text-muted-foreground">Scrape app reviews from Google Play Store.</p>
      </div>

      <div className="glass-card rounded-xl p-6">
        <h3 className="mb-4 text-lg font-semibold">New Scraping Job</h3>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="sm:col-span-2">
            <label className="mb-1.5 block text-sm text-muted-foreground">App ID</label>
            <Input
              placeholder="com.example.app"
              value={appId}
              onChange={(e) => setAppId(e.target.value)}
              className="bg-accent/50"
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">Language</label>
            <Input defaultValue="en" className="bg-accent/50" />
          </div>
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">Country</label>
            <Input defaultValue="us" className="bg-accent/50" />
          </div>
        </div>
        <Button onClick={handleStart} disabled={isRunning || !appId} className="mt-4 gradient-primary border-0">
          {isRunning ? (
            <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Scraping...</>
          ) : (
            <><Play className="mr-2 h-4 w-4" /> Start Scraping</>
          )}
        </Button>
      </div>

      <div className="glass-card rounded-xl p-5">
        <h3 className="mb-4 text-lg font-semibold">Scraping History</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-muted-foreground">
                <th className="py-3 pr-4 text-left font-medium">App ID</th>
                <th className="py-3 px-4 text-left font-medium">Status</th>
                <th className="py-3 px-4 text-left font-medium">Reviews</th>
                <th className="py-3 px-4 text-left font-medium">Date</th>
                <th className="py-3 pl-4 text-left font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {scrapingHistory.map((job) => (
                <tr key={job.id} className="border-b border-border/50 hover:bg-accent/50 transition-colors">
                  <td className="py-3 pr-4 font-mono text-xs">{job.appId}</td>
                  <td className="py-3 px-4">
                    {job.status === "completed" ? (
                      <span className="flex items-center gap-1 text-success text-xs"><CheckCircle2 className="h-3.5 w-3.5" /> Completed</span>
                    ) : (
                      <span className="flex items-center gap-1 text-warning text-xs"><Clock className="h-3.5 w-3.5" /> In Progress</span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-muted-foreground">{job.reviews.toLocaleString()}</td>
                  <td className="py-3 px-4 text-muted-foreground">{job.date}</td>
                  <td className="py-3 pl-4">
                    <Button variant="ghost" size="sm" className="h-7 text-xs">
                      <Download className="mr-1 h-3 w-3" /> CSV
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
