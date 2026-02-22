"use client";

import { Settings as SettingsIcon, Moon, Sun, Globe, Database, Bell } from "lucide-react";
import { Switch } from "@/components/ui/switch";

export default function SettingsPage() {
  return (
    <div className="max-w-2xl space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="mt-1 text-muted-foreground">Configure your ReviewNet preferences.</p>
      </div>

      <div className="glass-card rounded-xl p-5 space-y-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <SettingsIcon className="h-5 w-5 text-primary" /> General
        </h3>
        <div className="flex items-center justify-between py-2">
          <div>
            <p className="font-medium text-sm">Dark Mode</p>
            <p className="text-xs text-muted-foreground">Toggle between dark and light themes</p>
          </div>
          <Switch defaultChecked />
        </div>
        <div className="flex items-center justify-between py-2 border-t border-border/50">
          <div>
            <p className="font-medium text-sm">Language</p>
            <p className="text-xs text-muted-foreground">Application language</p>
          </div>
          <span className="text-sm text-muted-foreground">English</span>
        </div>
      </div>

      <div className="glass-card rounded-xl p-5 space-y-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Database className="h-5 w-5 text-primary" /> Data Settings
        </h3>
        <div className="flex items-center justify-between py-2">
          <div>
            <p className="font-medium text-sm">Auto-refresh</p>
            <p className="text-xs text-muted-foreground">Automatically refresh dashboard data</p>
          </div>
          <Switch defaultChecked />
        </div>
        <div className="flex items-center justify-between py-2 border-t border-border/50">
          <div>
            <p className="font-medium text-sm">Confidence Threshold</p>
            <p className="text-xs text-muted-foreground">Minimum confidence for analysis results</p>
          </div>
          <span className="text-sm text-muted-foreground">0.75</span>
        </div>
      </div>

      <div className="glass-card rounded-xl p-5 space-y-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Bell className="h-5 w-5 text-primary" /> Notifications
        </h3>
        <div className="flex items-center justify-between py-2">
          <div>
            <p className="font-medium text-sm">Email Notifications</p>
            <p className="text-xs text-muted-foreground">Receive alerts via email</p>
          </div>
          <Switch />
        </div>
        <div className="flex items-center justify-between py-2 border-t border-border/50">
          <div>
            <p className="font-medium text-sm">High Toxicity Alerts</p>
            <p className="text-xs text-muted-foreground">Alert when toxicity exceeds threshold</p>
          </div>
          <Switch defaultChecked />
        </div>
      </div>
    </div>
  );
}
