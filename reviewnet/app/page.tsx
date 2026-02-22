"use client";

import Link from "next/link";
import { TrendingUp, Sparkles, ShieldCheck, BarChart3, ArrowRight, Activity, Smile, Search, Heart, ShieldAlert, ThumbsUp, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0b] text-white selection:bg-primary/30 overflow-x-hidden">
      {/* Grid Background */}
      <div className="fixed inset-0 z-0 opacity-[0.03] pointer-events-none" 
           style={{ backgroundImage: "radial-gradient(#fff 1px, transparent 1px)", backgroundSize: "40px 40px" }} />
      
      {/* Decorative Blobs */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-primary/10 blur-[120px] pointer-events-none z-0" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-success/5 blur-[120px] pointer-events-none z-0" />

      {/* Nav */}
      <nav className="relative z-10 flex items-center justify-between px-6 py-6 max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl gradient-primary shadow-lg shadow-primary/20">
            <TrendingUp className="h-5 w-5 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
            ReviewNet
          </span>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-white/60">
          <a href="#features" className="hover:text-white transition-colors">Features</a>
          <a href="#workflow" className="hover:text-white transition-colors">Workflow</a>
          <a href="#models" className="hover:text-white transition-colors">Technology</a>
        </div>
        <Link href="/dashboard">
          <Button variant="ghost" className="rounded-full border border-white/10 bg-white/5 hover:bg-white/10 hover:text-white transition-all">
            Dashboard
          </Button>
        </Link>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-32 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 text-primary text-xs font-semibold mb-8 animate-fade-in">
          <Sparkles className="h-3 w-3" />
          <span>Advanced Sentiment & Toxicity Intelligence</span>
        </div>
        
        <h1 className="text-6xl md:text-8xl font-bold tracking-tight mb-8 animate-slide-up">
          Analyze Reviews <br />
          <span className="gradient-text">Like Never Before</span>
        </h1>
        
        <p className="max-w-2xl mx-auto text-lg md:text-xl text-white/50 mb-12 animate-slide-up" style={{ animationDelay: "0.1s" }}>
          The all-in-one intelligence toolkit for app review analysis. 
          Understand sentiment, detect toxicity, and uncover user emotions in seconds.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-slide-up" style={{ animationDelay: "0.2s" }}>
          <Link href="/dashboard">
            <Button size="lg" className="rounded-full px-8 py-6 text-md font-semibold gradient-primary hover:scale-105 transition-transform shadow-xl shadow-primary/20 border-0">
              Launch Console <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
          <Button size="lg" variant="ghost" className="rounded-full px-8 py-6 text-md font-semibold border border-white/10 bg-white/5 hover:bg-white/10">
            Watch Technical Demo
          </Button>
        </div>

        {/* Dashboard Preview Mockup - VIBRANT UPDATE */}
        <div className="mt-24 relative px-4 animate-slide-up group" style={{ animationDelay: "0.4s" }}>
          <div className="glass-card rounded-2xl p-2 border border-white/10 bg-white/5 backdrop-blur-xl overflow-hidden shadow-2xl relative z-10">
            <div className="rounded-xl border border-white/5 bg-[#0e0e11] overflow-hidden aspect-[16/9] flex flex-col">
              {/* Fake Dashboard Header */}
              <div className="h-12 border-b border-white/5 flex items-center justify-between px-4 bg-white/2">
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-destructive/40" />
                  <div className="w-2.5 h-2.5 rounded-full bg-warning/40" />
                  <div className="w-2.5 h-2.5 rounded-full bg-success/40" />
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1 w-32 rounded-full bg-white/5" />
                  <div className="h-5 w-5 rounded-full bg-primary/20 border border-primary/40 animate-pulse" />
                </div>
                <div className="flex gap-2">
                  <div className="w-6 h-6 rounded bg-white/5" />
                  <div className="w-6 h-6 rounded bg-white/5" />
                </div>
              </div>
              
              {/* Fake Dashboard Content */}
              <div className="flex-1 p-6 grid grid-cols-1 md:grid-cols-4 gap-6 overflow-hidden">
                {/* Sidebar Mockup */}
                <div className="hidden md:flex flex-col gap-4 border-r border-white/5 pr-4 mb-4">
                  {[TrendingUp, Activity, Smile, BarChart3, ShieldCheck].map((Icon, i) => (
                    <div key={i} className={`h-8 w-full rounded-lg flex items-center gap-3 px-2 ${i === 0 ? 'bg-primary/10 border border-primary/20' : 'bg-transparent'}`}>
                      <Icon className={`h-3.5 w-3.5 ${i === 0 ? 'text-primary' : 'text-white/20'}`} />
                      <div className={`h-1 flex-1 rounded-full ${i === 0 ? 'bg-primary/20' : 'bg-white/5'}`} />
                    </div>
                  ))}
                </div>

                {/* Main Content Mockup */}
                <div className="md:col-span-3 space-y-6">
                  {/* Metric Cards Row */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-3 rounded-xl bg-white/2 border border-white/5 relative group/card">
                       <div className="absolute top-2 right-2 h-4 w-4 text-success"><TrendingUp className="h-full w-full" /></div>
                       <div className="h-1 text-[10px] text-white/30 font-bold mb-3">SENTIMENT</div>
                       <div className="text-xl font-bold text-success tracking-tight">84.2%</div>
                       <div className="h-1 w-full bg-white/5 mt-3 rounded-full overflow-hidden">
                          <div className="h-full bg-success w-[84%]" />
                       </div>
                    </div>
                    <div className="p-3 rounded-xl bg-white/2 border border-white/5">
                       <div className="absolute top-2 right-2 h-4 w-4 text-primary"><Heart className="h-full w-full" /></div>
                       <div className="h-1 text-[10px] text-white/30 font-bold mb-3">TOP EMOTION</div>
                       <div className="text-xl font-bold text-white tracking-tight">Joy</div>
                    </div>
                    <div className="p-3 rounded-xl bg-white/2 border border-white/5">
                       <div className="absolute top-2 right-2 h-4 w-4 text-destructive"><ShieldAlert className="h-full w-full" /></div>
                       <div className="h-1 text-[10px] text-white/30 font-bold mb-3">TOXICITY</div>
                       <div className="text-xl font-bold text-destructive tracking-tight">0.12</div>
                    </div>
                  </div>

                  {/* Main Chart Mockup */}
                  <div className="h-44 rounded-xl bg-white/2 border border-white/5 p-4 relative overflow-hidden">
                    <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-primary/10 to-transparent" />
                    <div className="flex justify-between items-center mb-4">
                      <div className="h-1 text-[10px] text-white/30 font-bold">ANALYSIS OVER TIME</div>
                      <div className="flex gap-1">
                        <div className="w-2 h-2 rounded-sm bg-primary" />
                        <div className="w-2 h-2 rounded-sm bg-success" />
                        <div className="w-2 h-2 rounded-sm bg-destructive" />
                      </div>
                    </div>
                    <div className="w-full h-full flex items-end gap-2 px-2 pb-2">
                      {[60, 45, 90, 65, 80, 50, 85, 95, 60, 75, 55, 100].map((h, i) => (
                        <div key={i} className="flex-1 relative group/bar">
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 opacity-0 group-hover/bar:opacity-100 transition-opacity whitespace-nowrap bg-white/10 px-1 rounded text-[8px]">
                            {h}%
                          </div>
                          <div className={`w-full rounded-t-sm transition-all duration-500 hover:brightness-125 ${i % 2 === 0 ? 'bg-primary/40' : 'bg-success/40'}`} 
                               style={{ height: `${h}%` }} />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Activity List Mockup */}
                  <div className="space-y-2">
                    {[
                      { icon: ThumbsUp, color: 'text-success', bg: 'bg-success/10', text: "New positive review pulse from Spotify" },
                      { icon: ShieldAlert, color: 'text-destructive', bg: 'bg-destructive/10', text: "Toxicity threshold exceeded in WhatsApp" },
                      { icon: MessageSquare, color: 'text-primary', bg: 'bg-primary/10', text: "Bilingual analysis completed for TikTok" }
                    ].map((item, i) => (
                      <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-white/2 border border-white/5">
                        <div className={`p-1.5 rounded ${item.bg} ${item.color}`}>
                          <item.icon className="h-3 w-3" />
                        </div>
                        <div className="text-[11px] text-white/60 truncate font-medium">{item.text}</div>
                        <div className="ml-auto text-[9px] text-white/20">{i+1}m ago</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Enhanced Glow Effects */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-primary/20 blur-[150px] -z-10 group-hover:bg-primary/30 transition-colors" />
          <div className="absolute top-[-10%] right-[10%] w-32 h-32 bg-success/20 blur-[60px] -z-10" />
          <div className="absolute bottom-[-10%] left-[10%] w-32 h-32 bg-destructive/10 blur-[60px] -z-10" />
        </div>
      </section>

      {/* Workflow Section - Better Visualization */}
      <section id="workflow" className="relative z-10 max-w-7xl mx-auto px-6 py-32 border-t border-white/[0.02]">
        <div className="text-center mb-20">
          <h2 className="text-4xl font-bold mb-4">How It Works</h2>
          <p className="text-white/50">From raw feedback to deep business intelligence.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-12 relative">
          {[
            { step: "01", title: "Scrape", desc: "Real-time extraction from Play Store across all regions.", color: "primary" },
            { step: "02", title: "Analyze", desc: "BERT & RoBERTa models classify sentiment and emotions.", color: "success" },
            { step: "03", title: "Decision", desc: "Actionable data visualize through custom dashboards.", color: "warning" }
          ].map((item, i) => (
            <div key={i} className="glass-card p-8 rounded-3xl border border-white/10 bg-white/5 hover:translate-y-[-5px] transition-transform">
               <div className={`text-5xl font-black mb-6 opacity-10 text-${item.color}`}>{item.step}</div>
               <h3 className="text-xl font-bold mb-3 uppercase tracking-wider">{item.title}</h3>
               <p className="text-white/50 text-sm leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section id="features" className="relative z-10 max-w-7xl mx-auto px-6 py-32">
        <div className="text-center mb-20">
          <h2 className="text-4xl font-bold mb-4">Deep Insights, Simplified</h2>
          <p className="text-white/50">Everything you need to turn user feedback into actionable data.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          <div className="glass-card rounded-2xl p-8 border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-colors group">
            <div className="h-12 w-12 rounded-xl bg-success/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Activity className="h-6 w-6 text-success" />
            </div>
            <h3 className="text-xl font-bold mb-3 tracking-tight">Sentiment Analysis</h3>
            <p className="text-white/50 text-sm leading-relaxed">
              Understand the overall mood of your users with high-precision sentiment classification including positive, neutral, and negative.
            </p>
          </div>

          <div className="glass-card rounded-2xl p-8 border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-colors group">
            <div className="h-12 w-12 rounded-xl bg-warning/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Smile className="h-6 w-6 text-warning" />
            </div>
            <h3 className="text-xl font-bold mb-3 tracking-tight">Emotion Detection</h3>
            <p className="text-white/50 text-sm leading-relaxed">
              Go beyond simple sentiment. Detect specific emotions like Joy, Anger, Sadness, and Surprise to truly understand user intent.
            </p>
          </div>

          <div className="glass-card rounded-2xl p-8 border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-colors group">
            <div className="h-12 w-12 rounded-xl bg-destructive/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <ShieldCheck className="h-6 w-6 text-destructive" />
            </div>
            <h3 className="text-xl font-bold mb-3 tracking-tight">Toxicity Shield</h3>
            <p className="text-white/50 text-sm leading-relaxed">
              Automatically flag toxic, hateful, or abusive reviews. Maintain a healthy community and protect your brand reputation.
            </p>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-10 bg-white/[0.02] border-y border-white/5 py-24">
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-12 text-center">
          <div>
            <div className="text-4xl font-bold mb-2 tracking-tighter">99.2%</div>
            <div className="text-white/40 text-[10px] uppercase tracking-widest font-bold">Inference Accuracy</div>
          </div>
          <div>
            <div className="text-4xl font-bold mb-2 tracking-tighter">1M+</div>
            <div className="text-white/40 text-[10px] uppercase tracking-widest font-bold">Reviews Processed</div>
          </div>
          <div>
            <div className="text-4xl font-bold mb-2 tracking-tighter">24/7</div>
            <div className="text-white/40 text-[10px] uppercase tracking-widest font-bold">Real-time Scraping</div>
          </div>
          <div>
            <div className="text-4xl font-bold mb-2 tracking-tighter">12ms</div>
            <div className="text-white/40 text-[10px] uppercase tracking-widest font-bold">API Latency</div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="relative z-10 max-w-7xl mx-auto px-6 py-32 text-center">
        <div className="glass-card rounded-[2.5rem] p-16 border border-white/10 bg-gradient-to-br from-primary/10 via-transparent to-primary/5 overflow-hidden relative shadow-2xl">
          <div className="relative z-10">
            <h2 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight italic underline decoration-primary/30 underline-offset-8">Ready to transform your research?</h2>
            <p className="text-white/50 mb-10 max-w-xl mx-auto text-lg underline-offset-4 decoration-white/10 decoration-1 underline">
              Join the future of app intelligence. Start your free analysis today.
            </p>
            <Link href="/dashboard">
              <Button size="lg" className="rounded-full px-12 py-7 text-lg font-bold gradient-primary shadow-2xl shadow-primary/30 border-0 hover:scale-105 transition-transform">
                Launch Dashboard Now
              </Button>
            </Link>
          </div>
          {/* Background decoration */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/10 blur-[120px] -z-10" />
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 py-12 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:row items-center justify-between gap-6">
          <div className="flex items-center gap-2 grayscale brightness-200 opacity-50">
            <TrendingUp className="h-5 w-5" />
            <span className="font-bold">ReviewNet AI</span>
          </div>
          <div className="text-white/30 text-xs font-mono">
            BUILD_VERSION_2026.02
          </div>
          <div className="flex gap-6 text-white/30 text-sm">
            <a href="#" className="hover:text-white">Privacy</a>
            <a href="#" className="hover:text-white">Terms</a>
            <a href="https://github.com/aabubokarr/reviewnet" className="hover:text-white">Source</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
