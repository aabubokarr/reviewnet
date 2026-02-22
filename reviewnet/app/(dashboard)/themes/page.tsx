"use client";

import { Layers } from "lucide-react";
import { themeDistribution, sentimentByApp } from "@/lib/mock-data";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";

const THEME_COLORS = [
  "hsl(239,84%,67%)", "hsl(263,70%,50%)", "hsl(160,84%,39%)",
  "hsl(217,91%,60%)", "hsl(38,92%,50%)", "hsl(0,84%,60%)", "hsl(280,60%,50%)",
];

export default function ThemesPage() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold">Theme Analysis</h1>
        <p className="mt-1 text-muted-foreground">Review categorization by theme and topic.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="glass-card rounded-xl p-5">
          <h3 className="mb-4 text-lg font-semibold">Theme Distribution</h3>
          <ResponsiveContainer width="100%" height={350}>
            <PieChart>
              <Pie data={themeDistribution} cx="50%" cy="50%" outerRadius={120} dataKey="count" nameKey="theme" paddingAngle={2}>
                {themeDistribution.map((_, i) => <Cell key={i} fill={THEME_COLORS[i]} />)}
              </Pie>
              <Tooltip contentStyle={{ background:"hsl(217,33%,17%)", border:"1px solid hsl(217,33%,30%)", borderRadius:"8px", color:"hsl(210,40%,98%)" }} />
              <Legend formatter={(v) => <span className="text-sm text-muted-foreground">{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-card rounded-xl p-5">
          <h3 className="mb-4 text-lg font-semibold">Themes by Volume</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={themeDistribution} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,22%)" />
              <XAxis type="number" stroke="hsl(215,20%,65%)" fontSize={12} />
              <YAxis dataKey="theme" type="category" stroke="hsl(215,20%,65%)" fontSize={12} width={120} />
              <Tooltip contentStyle={{ background:"hsl(217,33%,17%)", border:"1px solid hsl(217,33%,30%)", borderRadius:"8px", color:"hsl(210,40%,98%)" }} />
              <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                {themeDistribution.map((_, i) => <Cell key={i} fill={THEME_COLORS[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {themeDistribution.map((t, i) => (
          <div key={t.theme} className="glass-card rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-3 w-3 rounded-full" style={{ background: THEME_COLORS[i] }} />
              <h4 className="font-semibold text-sm">{t.theme}</h4>
            </div>
            <p className="text-2xl font-bold">{t.count.toLocaleString()}</p>
            <p className="text-xs text-muted-foreground">reviews categorized</p>
          </div>
        ))}
      </div>
    </div>
  );
}
