"use client";

import {
    BarChart3, Heart, ShieldAlert, TrendingUp, Activity, Sparkles,
  } from "lucide-react";
  import MetricCard from "@/components/dashboard/MetricCard";
  import {
    dashboardStats, sentimentOverTime, sentimentDistribution,
    topApps, emotionDistribution, recentActivity,
  } from "@/lib/mock-data";
  import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, BarChart, Bar, Legend,
  } from "recharts";
  
  const SENTIMENT_COLORS = ["hsl(160, 84%, 39%)", "hsl(217, 91%, 60%)", "hsl(0, 84%, 60%)"];
  
  const activityTypeColors: Record<string, string> = {
    success: "bg-success",
    info: "bg-info",
    danger: "bg-destructive",
  };
  
  export default function Dashboard() {
    return (
      <div className="space-y-6 animate-fade-in">
        <div>
          <h1 className="text-3xl font-bold">
            Welcome back, <span className="gradient-text">Admin</span>
          </h1>
          <p className="mt-1 text-muted-foreground">
            Here's what's happening with your app reviews today.
          </p>
        </div>
  
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          <MetricCard title="Total Reviews" value={dashboardStats.totalReviews.toLocaleString()} trend={dashboardStats.totalReviewsTrend} icon={BarChart3} />
          <MetricCard title="Positive" value={`${dashboardStats.positivePct}%`} trend={3.2} icon={TrendingUp} variant="success" />
          <MetricCard title="Negative" value={`${dashboardStats.negativePct}%`} trend={-2.1} icon={TrendingUp} variant="danger" />
          <MetricCard title="Avg Toxicity" value={dashboardStats.avgToxicity.toFixed(2)} trend={-5.4} icon={ShieldAlert} variant="warning" />
          <MetricCard title="Top Emotion" value={dashboardStats.topEmotion} subtitle="33.9% of reviews" icon={Heart} variant="info" />
          <MetricCard title="Active Apps" value={dashboardStats.activeApps} trend={8.3} icon={Activity} />
        </div>
  
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="glass-card rounded-xl p-5 lg:col-span-2">
            <h3 className="mb-4 text-lg font-semibold">Sentiment Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={sentimentOverTime}>
                <defs>
                  <linearGradient id="posGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(160, 84%, 39%)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(160, 84%, 39%)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="negGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 22%)" />
                <XAxis dataKey="month" stroke="hsl(215, 20%, 65%)" fontSize={12} />
                <YAxis stroke="hsl(215, 20%, 65%)" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    background: "hsl(217, 33%, 17%)",
                    border: "1px solid hsl(217, 33%, 30%)",
                    borderRadius: "8px",
                    color: "hsl(210, 40%, 98%)",
                  }}
                />
                <Area type="monotone" dataKey="positive" stroke="hsl(160, 84%, 39%)" fill="url(#posGrad)" strokeWidth={2} />
                <Area type="monotone" dataKey="neutral" stroke="hsl(217, 91%, 60%)" fill="transparent" strokeWidth={2} />
                <Area type="monotone" dataKey="negative" stroke="hsl(0, 84%, 60%)" fill="url(#negGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
  
          <div className="glass-card rounded-xl p-5">
            <h3 className="mb-4 text-lg font-semibold">Sentiment Split</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={sentimentDistribution}
                  cx="50%" cy="50%"
                  innerRadius={60} outerRadius={100}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {sentimentDistribution.map((_, i) => (
                    <Cell key={i} fill={SENTIMENT_COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "hsl(217, 33%, 17%)",
                    border: "1px solid hsl(217, 33%, 30%)",
                    borderRadius: "8px",
                    color: "hsl(210, 40%, 98%)",
                  }}
                />
                <Legend
                  formatter={(value) => <span className="text-sm text-muted-foreground">{value}</span>}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
  
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="glass-card rounded-xl p-5">
            <h3 className="mb-4 text-lg font-semibold">Top Apps by Volume</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={topApps} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(217, 33%, 22%)" />
                <XAxis type="number" stroke="hsl(215, 20%, 65%)" fontSize={12} />
                <YAxis dataKey="app" type="category" stroke="hsl(215, 20%, 65%)" fontSize={12} width={80} />
                <Tooltip
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
              <Sparkles className="h-5 w-5 text-primary" /> Recent Activity
            </h3>
            <div className="space-y-4">
              {recentActivity.map((item) => (
                <div key={item.id} className="flex items-center gap-3">
                  <div className={`h-2.5 w-2.5 rounded-full ${activityTypeColors[item.type]}`} />
                  <div className="flex-1">
                    <p className="text-sm font-medium">{item.action}</p>
                    <p className="text-xs text-muted-foreground">{item.app}</p>
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
