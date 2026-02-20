"use client";

import { ShieldAlert, ShieldCheck, AlertTriangle } from "lucide-react";
import MetricCard from "@/components/dashboard/MetricCard";
import { toxicityByApp, sampleReviews } from "@/lib/mock-data";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { cn } from "@/lib/utils";

export default function ToxicityPage() {
  const highToxReviews = sampleReviews.filter((r) => r.toxicity > 0.15);

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold">Toxicity Detection</h1>
        <p className="mt-1 text-muted-foreground">Monitor and manage toxic content across your apps.</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-4">
        <MetricCard title="Avg Toxicity" value="0.12" icon={ShieldAlert} variant="warning" />
        <MetricCard title="Low Toxicity" value="18,600" subtitle="< 0.3 score" icon={ShieldCheck} variant="success" />
        <MetricCard title="Medium Toxicity" value="4,150" subtitle="0.3 – 0.7" icon={AlertTriangle} variant="warning" />
        <MetricCard title="High Toxicity" value="2,103" subtitle="> 0.7 score" icon={ShieldAlert} variant="danger" />
      </div>

      <div className="glass-card rounded-xl p-5">
        <h3 className="mb-4 text-lg font-semibold">Toxicity by App</h3>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={toxicityByApp}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,22%)" />
            <XAxis dataKey="app" stroke="hsl(215,20%,65%)" fontSize={12} />
            <YAxis stroke="hsl(215,20%,65%)" fontSize={12} />
            <Tooltip contentStyle={{ background:"hsl(217,33%,17%)", border:"1px solid hsl(217,33%,30%)", borderRadius:"8px", color:"hsl(210,40%,98%)" }} />
            <Legend formatter={(v) => <span className="text-sm text-muted-foreground capitalize">{v}</span>} />
            <Bar dataKey="low" stackId="a" fill="hsl(160,84%,39%)" />
            <Bar dataKey="medium" stackId="a" fill="hsl(38,92%,50%)" />
            <Bar dataKey="high" stackId="a" fill="hsl(0,84%,60%)" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Flagged Reviews */}
      <div className="glass-card rounded-xl p-5">
        <h3 className="mb-4 text-lg font-semibold text-destructive flex items-center gap-2">
          <ShieldAlert className="h-5 w-5" /> Flagged Reviews
        </h3>
        <div className="space-y-3">
          {highToxReviews.map((r) => (
            <div key={r.id} className="rounded-lg border border-destructive/20 bg-destructive/5 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">{r.app}</span>
                <span className="text-xs rounded-full bg-destructive/15 text-destructive px-2 py-0.5 font-medium">
                  Score: {r.toxicity.toFixed(2)}
                </span>
              </div>
              <p className="text-sm text-muted-foreground">{r.content}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
