"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { TrendingUp, Home } from "lucide-react";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-[#0a0a0b] text-white p-6 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl gradient-primary mb-8 shadow-2xl shadow-primary/20">
        <TrendingUp className="h-8 w-8 text-white" />
      </div>
      <h1 className="text-9xl font-bold tracking-tighter gradient-text mb-4">404</h1>
      <h2 className="text-2xl font-semibold mb-6">Page Not Found</h2>
      <p className="max-w-md text-white/50 mb-10">
        The page you are looking for doesn't exist or has been moved. 
        Don't worry, you can always head back to the dashboard.
      </p>
      <div className="flex gap-4">
        <Link href="/dashboard">
          <Button size="lg" className="rounded-full px-8 gradient-primary border-0">
            <Home className="mr-2 h-4 w-4" /> Go to Dashboard
          </Button>
        </Link>
        <Link href="/">
          <Button size="lg" variant="ghost" className="rounded-full px-8 border border-white/10 bg-white/5">
            Back to Home
          </Button>
        </Link>
      </div>
    </div>
  );
}
