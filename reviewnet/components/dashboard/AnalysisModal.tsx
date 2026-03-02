"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ClipboardList, Sparkles, Upload, ChevronLeft } from "lucide-react";

export function AnalysisModal() {
  const [view, setView] = useState<"selection" | "automated">("selection");

  return (
    <Dialog onOpenChange={(open) => !open && setView("selection")}>
      <DialogTrigger asChild>
        <Button className="gradient-primary">
          Analysis
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="flex items-center gap-2">
            {view === "automated" && (
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8"
                onClick={() => setView("selection")}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
            )}
            <DialogTitle>
              {view === "selection" ? "Select Analysis Mode" : "Upload Documents"}
            </DialogTitle>
          </div>
          <DialogDescription>
            {view === "selection" 
              ? "Choose how you want to analyze the reviews." 
              : "Upload documents for automated AI analysis."}
          </DialogDescription>
        </DialogHeader>

        {view === "selection" ? (
          <div className="grid grid-cols-2 gap-4 py-4">
            <Link href="/analysis/manual" className="contents">
              <button className="flex flex-col items-center justify-center gap-3 rounded-xl border border-border bg-accent/50 p-6 transition-all hover:bg-accent hover:shadow-md group text-left">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  <ClipboardList className="h-6 w-6" />
                </div>
                <div className="text-center">
                  <p className="font-semibold">Manual</p>
                  <p className="text-xs text-muted-foreground whitespace-nowrap">Human centric analysis</p>
                </div>
              </button>
            </Link>
            
            <button 
              onClick={() => setView("automated")}
              className="flex flex-col items-center justify-center gap-3 rounded-xl border border-border bg-accent/50 p-6 transition-all hover:bg-accent hover:shadow-md group"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                <Sparkles className="h-6 w-6" />
              </div>
              <div className="text-center">
                <p className="font-semibold">Automated</p>
                <p className="text-xs text-muted-foreground whitespace-nowrap">AI-powered insights</p>
              </div>
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-4 py-8 border-2 border-dashed border-border rounded-xl bg-accent/5 mt-4">
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
              <Upload className="h-6 w-6" />
            </div>
            <div className="text-center">
              <p className="font-medium">Click to upload or drag and drop</p>
              <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, or TXT (max. 10MB)</p>
            </div>
            <Button size="sm" className="mt-2">Select Files</Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
