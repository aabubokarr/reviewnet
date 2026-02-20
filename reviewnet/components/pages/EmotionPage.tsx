"use client";

import { Heart, Smile, Frown } from "lucide-react";
import MetricCard from "@/components/dashboard/MetricCard";
import { emotionDistribution } from "@/lib/mock-data";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const EMOTION_COLORS = [
  "hsl(160,84%,39%)", "hsl(217,91%,60%)", "hsl(239,84%,67%)", "hsl(263,70%,50%)",
  "hsl(38,92%,50%)", "hsl(0,84%,60%)", "hsl(350,80%,50%)", "hsl(280,60%,50%)",
];

export default function EmotionPage() {
  const top3 = emotionDistribution.slice(0, 3);

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold">Emotion Detection</h1>
        <p className="mt-1 text-muted-foreground">Understand the emotional landscape of your reviews.</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <MetricCard title={`#1 ${top3[0].emotion}`} value={`${top3[0].pct}%`} subtitle={`${top3[0].count.toLocaleString()} reviews`} icon={Smile} variant="success" />
        <MetricCard title={`#2 ${top3[1].emotion}`} value={`${top3[1].pct}%`} subtitle={`${top3[1].count.toLocaleString()} reviews`} icon={Heart} variant="info" />
        <MetricCard title={`#3 ${top3[2].emotion}`} value={`${top3[2].pct}%`} subtitle={`${top3[2].count.toLocaleString()} reviews`} icon={Frown} variant="default" />
      </div>

      <div className="glass-card rounded-xl p-5">
        <h3 className="mb-4 text-lg font-semibold">Emotion Distribution</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={emotionDistribution} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,22%)" />
            <XAxis type="number" stroke="hsl(215,20%,65%)" fontSize={12} />
            <YAxis dataKey="emotion" type="category" stroke="hsl(215,20%,65%)" fontSize={12} width={100} />
            <Tooltip contentStyle={{ background:"hsl(217,33%,17%)", border:"1px solid hsl(217,33%,30%)", borderRadius:"8px", color:"hsl(210,40%,98%)" }} />
            <Bar dataKey="count" radius={[0, 6, 6, 0]}>
              {emotionDistribution.map((_, i) => (
                <Cell key={i} fill={EMOTION_COLORS[i]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Emotion Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {emotionDistribution.map((e, i) => (
          <div key={e.emotion} className="glass-card rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-3 w-3 rounded-full" style={{ background: EMOTION_COLORS[i] }} />
              <h4 className="font-semibold">{e.emotion}</h4>
            </div>
            <p className="text-2xl font-bold">{e.count.toLocaleString()}</p>
            <p className="text-xs text-muted-foreground">{e.pct}% of total</p>
          </div>
        ))}
      </div>
    </div>
  );
}
