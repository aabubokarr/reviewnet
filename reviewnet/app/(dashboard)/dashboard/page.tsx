"use client";

import {
    BarChart3, Heart, ShieldAlert, TrendingUp, Activity, Sparkles, Cloud
  } from "lucide-react";
  import MetricCard from "@/components/dashboard/MetricCard";
  import {
    dashboardStats, sentimentOverTime, sentimentDistribution,
    topThemes, emotionDistribution, recentActivity, wordSets,
    themeDistribution, toxicityDistribution
  } from "@/lib/mock-data";
  import { AnalysisModal } from "@/components/dashboard/AnalysisModal";
  import { WordCloud } from "@/components/dashboard/WordCloud";
  import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, BarChart, Bar, Legend,
  } from "recharts";
  
  const SENTIMENT_COLORS = ["hsl(160, 84%, 39%)", "hsl(217, 91%, 60%)", "hsl(0, 84%, 60%)"];
  const THEME_COLORS = [
    "hsl(239, 84%, 67%)", "hsl(263, 70%, 50%)", "hsl(160, 84%, 39%)",
    "hsl(217, 91%, 60%)", "hsl(38, 92%, 50%)", "hsl(0, 84%, 60%)",
  ];
  
  const activityTypeColors: Record<string, string> = {
    success: "bg-success",
    info: "bg-info",
    danger: "bg-destructive",
  };
  
  export default function Dashboard() {
    return (
      <div className="space-y-6 animate-fade-in pb-10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">
              Welcome back, <span className="gradient-text">Admin</span>
            </h1>
            <p className="mt-1 text-muted-foreground">
              Here's what's happening with your thematic analysis today.
            </p>
          </div>
          <AnalysisModal />
        </div>
  
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          <MetricCard title="Total Reviews" value={dashboardStats.totalReviews.toLocaleString()} trend={dashboardStats.totalReviewsTrend} icon={BarChart3} />
          <MetricCard title="Negative" value={`${dashboardStats.negativePct}%`} trend={-2.1} icon={TrendingUp} variant="danger" />
          <MetricCard title="Avg Toxicity" value={dashboardStats.avgToxicity.toFixed(2)} trend={-5.4} icon={ShieldAlert} variant="warning" />
          <MetricCard title="Top Emotion" value={dashboardStats.topEmotion} subtitle="33.9% of reviews" icon={Heart} variant="info" />
          <MetricCard title="Active Themes" value={topThemes.length} trend={8.3} icon={Activity} />
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {/* Theme Distribution */}
          <div className="glass-card rounded-xl p-5 flex flex-col">
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">Theme Distribution</h3>
            <div className="flex-1 min-h-[220px] flex flex-col items-center justify-center">
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie
                    data={themeDistribution}
                    outerRadius={70}
                    paddingAngle={0}
                    stroke="none"
                    dataKey="count"
                  >
                    {themeDistribution.map((_, i) => (
                      <Cell key={i} fill={THEME_COLORS[i % THEME_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip cursor={false} />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 flex flex-wrap justify-center gap-x-4 gap-y-1">
                {themeDistribution.map((item, i) => (
                  <div key={item.theme} className="flex items-center gap-1.5">
                    <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: THEME_COLORS[i % THEME_COLORS.length] }} />
                    <span className="text-[10px] font-medium text-muted-foreground">{item.theme}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sentiment Distribution */}
          <div className="glass-card rounded-xl p-5 flex flex-col">
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">Sentiment Split</h3>
            <div className="flex-1 min-h-[220px] flex flex-col items-center justify-center">
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie
                    data={sentimentDistribution}
                    outerRadius={70}
                    paddingAngle={0}
                    stroke="none"
                    dataKey="value"
                  >
                    {sentimentDistribution.map((_, i) => (
                      <Cell key={i} fill={SENTIMENT_COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip cursor={false} />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 flex justify-center gap-6">
                {sentimentDistribution.map((item, i) => (
                  <div key={item.name} className="flex items-center gap-1.5">
                    <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: SENTIMENT_COLORS[i] }} />
                    <span className="text-[10px] font-medium text-muted-foreground">{item.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Emotion Distribution */}
          <div className="glass-card rounded-xl p-5 flex flex-col">
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">Top Emotions</h3>
            <div className="flex-1 min-h-[220px] flex flex-col items-center justify-center">
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie
                    data={emotionDistribution.slice(0, 5)}
                    outerRadius={70}
                    paddingAngle={0}
                    stroke="none"
                    dataKey="count"
                  >
                    {emotionDistribution.slice(0, 5).map((_, i) => (
                      <Cell key={i} fill={THEME_COLORS[(i + 2) % THEME_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip cursor={false} />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 flex flex-wrap justify-center gap-x-4 gap-y-1">
                {emotionDistribution.slice(0, 5).map((item, i) => (
                  <div key={item.emotion} className="flex items-center gap-1.5">
                    <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: THEME_COLORS[(i + 2) % THEME_COLORS.length] }} />
                    <span className="text-[10px] font-medium text-muted-foreground">{item.emotion}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Toxicity Distribution */}
          <div className="glass-card rounded-xl p-5 flex flex-col">
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">Toxicity Levels</h3>
            <div className="flex-1 min-h-[220px] flex flex-col items-center justify-center">
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie
                    data={toxicityDistribution}
                    outerRadius={70}
                    paddingAngle={0}
                    stroke="none"
                    dataKey="value"
                  >
                    <Cell fill="hsl(160, 84%, 39%)" />
                    <Cell fill="hsl(38, 92%, 50%)" />
                    <Cell fill="hsl(0, 84%, 60%)" />
                  </Pie>
                  <Tooltip cursor={false} />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 flex justify-center gap-6">
                {toxicityDistribution.map((item, i) => (
                  <div key={item.name} className="flex items-center gap-1.5">
                    <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: i === 0 ? "hsl(160, 84%, 39%)" : i === 1 ? "hsl(38, 92%, 50%)" : "hsl(0, 84%, 60%)" }} />
                    <span className="text-[10px] font-medium text-muted-foreground">{item.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Word Clouds Row */}
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="glass-card rounded-xl p-5 border-success/20">
            <h3 className="mb-2 text-sm font-semibold text-success flex items-center gap-2 uppercase tracking-wider">
              <Cloud className="h-4 w-4" /> Positive Buzz
            </h3>
            <WordCloud words={wordSets.positive} color="hsl(160, 84%, 39%)" />
          </div>
          <div className="glass-card rounded-xl p-5 border-info/20">
            <h3 className="mb-2 text-sm font-semibold text-info flex items-center gap-2 uppercase tracking-wider">
              <Cloud className="h-4 w-4" /> Neutral Mentions
            </h3>
            <WordCloud words={wordSets.neutral} color="hsl(217, 91%, 60%)" />
          </div>
          <div className="glass-card rounded-xl p-5 border-destructive/20">
            <h3 className="mb-2 text-sm font-semibold text-destructive flex items-center gap-2 uppercase tracking-wider">
              <Cloud className="h-4 w-4" /> Negative Pain Points
            </h3>
            <WordCloud words={wordSets.negative} color="hsl(0, 84%, 60%)" />
          </div>
        </div>
  
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="glass-card rounded-xl p-5">
            <h3 className="mb-4 text-lg font-semibold">Top Themes by Volume</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={topThemes} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 22%)" />
                <XAxis type="number" stroke="hsl(215, 20%, 65%)" fontSize={12} />
                <YAxis dataKey="theme" type="category" stroke="hsl(215, 20%, 65%)" fontSize={12} width={100} />
                <Tooltip
                  cursor={false}
                  contentStyle={{
                    background: "hsl(217, 33%, 17%)",
                    border: "1px solid hsl(217, 33%, 30%)",
                    borderRadius: "8px",
                    color: "hsl(210, 40%, 98%)",
                  }}
                />
                <Bar dataKey="reviews" fill="hsl(239, 84%, 67%)" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
  
          <div className="glass-card rounded-xl p-5">
            <h3 className="mb-4 text-lg font-semibold flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" /> Recent Insights
            </h3>
            <div className="space-y-4">
              {recentActivity.map((item) => (
                <div key={item.id} className="flex items-center gap-3">
                  <div className={`h-2.5 w-2.5 rounded-full ${activityTypeColors[item.type]}`} />
                  <div className="flex-1">
                    <p className="text-sm font-medium">{item.action}</p>
                    <p className="text-xs text-muted-foreground">{item.theme}</p>
                  </div>
                  <span className="text-xs text-muted-foreground">{item.time}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }
