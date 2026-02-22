"use client";

import { TrendingUp, ThumbsUp, ThumbsDown, Minus } from "lucide-react";
import MetricCard from "@/components/dashboard/MetricCard";
import {
  dashboardStats, sentimentOverTime, sentimentDistribution,
  sentimentByApp, sampleReviews,
} from "@/lib/mock-data";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar, Legend,
} from "recharts";
import { cn } from "@/lib/utils";

const COLORS = ["hsl(160, 84%, 39%)", "hsl(217, 91%, 60%)", "hsl(0, 84%, 60%)"];

const sentimentBadge: Record<string, string> = {
  Positive: "bg-success/15 text-success",
  Neutral: "bg-info/15 text-info",
  Negative: "bg-destructive/15 text-destructive",
};

export default function SentimentPage() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold">Sentiment Analysis</h1>
        <p className="mt-1 text-muted-foreground">Comprehensive sentiment breakdown of app reviews.</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <MetricCard title="Positive Reviews" value={`${dashboardStats.positivePct}%`} subtitle="15,906 reviews" trend={3.2} icon={ThumbsUp} variant="success" />
        <MetricCard title="Neutral Reviews" value={`${dashboardStats.neutralPct}%`} subtitle="4,474 reviews" icon={Minus} variant="info" />
        <MetricCard title="Negative Reviews" value={`${dashboardStats.negativePct}%`} subtitle="4,473 reviews" trend={-2.1} icon={ThumbsDown} variant="danger" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="glass-card rounded-xl p-5 lg:col-span-2">
          <h3 className="mb-4 text-lg font-semibold">Sentiment Over Time</h3>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={sentimentOverTime}>
              <defs>
                <linearGradient id="sg1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(160,84%,39%)" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="hsl(160,84%,39%)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,22%)" />
              <XAxis dataKey="month" stroke="hsl(215,20%,65%)" fontSize={12}/>
              <YAxis stroke="hsl(215,20%,65%)" fontSize={12}/>
              <Tooltip contentStyle={{ background:"hsl(217,33%,17%)", border:"1px solid hsl(217,33%,30%)", borderRadius:"8px", color:"hsl(210,40%,98%)" }}/>
              <Area type="monotone" dataKey="positive" stroke="hsl(160,84%,39%)" fill="url(#sg1)" strokeWidth={2}/>
              <Area type="monotone" dataKey="neutral" stroke="hsl(217,91%,60%)" fill="transparent" strokeWidth={2}/>
              <Area type="monotone" dataKey="negative" stroke="hsl(0,84%,60%)" fill="transparent" strokeWidth={2}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-card rounded-xl p-5">
          <h3 className="mb-4 text-lg font-semibold">Distribution</h3>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie data={sentimentDistribution} cx="50%" cy="50%" innerRadius={55} outerRadius={95} paddingAngle={4} dataKey="value">
                {sentimentDistribution.map((_, i) => <Cell key={i} fill={COLORS[i]}/>)}
              </Pie>
              <Tooltip contentStyle={{ background:"hsl(217,33%,17%)", border:"1px solid hsl(217,33%,30%)", borderRadius:"8px", color:"hsl(210,40%,98%)" }}/>
              <Legend formatter={(v) => <span className="text-sm text-muted-foreground">{v}</span>}/>
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-card rounded-xl p-5">
        <h3 className="mb-4 text-lg font-semibold">Sentiment by App</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={sentimentByApp}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,22%)" />
            <XAxis dataKey="app" stroke="hsl(215,20%,65%)" fontSize={12}/>
            <YAxis stroke="hsl(215,20%,65%)" fontSize={12}/>
            <Tooltip contentStyle={{ background:"hsl(217,33%,17%)", border:"1px solid hsl(217,33%,30%)", borderRadius:"8px", color:"hsl(210,40%,98%)" }}/>
            <Legend formatter={(v) => <span className="text-sm text-muted-foreground capitalize">{v}</span>}/>
            <Bar dataKey="positive" fill="hsl(160,84%,39%)" radius={[4,4,0,0]}/>
            <Bar dataKey="neutral" fill="hsl(217,91%,60%)" radius={[4,4,0,0]}/>
            <Bar dataKey="negative" fill="hsl(0,84%,60%)" radius={[4,4,0,0]}/>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="glass-card rounded-xl p-5">
        <h3 className="mb-4 text-lg font-semibold">Recent Reviews</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-muted-foreground">
                <th className="py-3 pr-4 text-left font-medium">Review</th>
                <th className="py-3 px-4 text-left font-medium">App</th>
                <th className="py-3 px-4 text-left font-medium">Sentiment</th>
                <th className="py-3 px-4 text-left font-medium">Confidence</th>
                <th className="py-3 pl-4 text-left font-medium">Date</th>
              </tr>
            </thead>
            <tbody>
              {sampleReviews.map((r) => (
                <tr key={r.id} className="border-b border-border/50 hover:bg-accent/50 transition-colors">
                  <td className="py-3 pr-4 max-w-xs truncate">{r.content}</td>
                  <td className="py-3 px-4 text-muted-foreground">{r.app}</td>
                  <td className="py-3 px-4">
                    <span className={cn("rounded-full px-2.5 py-0.5 text-xs font-medium", sentimentBadge[r.sentiment])}>
                      {r.sentiment}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-muted-foreground">{(r.confidence * 100).toFixed(0)}%</td>
                  <td className="py-3 pl-4 text-muted-foreground">{r.date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
