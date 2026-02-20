import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { LucideIcon, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: number;
  icon: LucideIcon;
  variant?: "default" | "success" | "danger" | "info" | "warning";
  children?: ReactNode;
}

const variantStyles = {
  default: "from-primary/20 to-primary/5",
  success: "from-success/20 to-success/5",
  danger: "from-destructive/20 to-destructive/5",
  info: "from-info/20 to-info/5",
  warning: "from-warning/20 to-warning/5",
};

const iconVariantStyles = {
  default: "gradient-primary",
  success: "gradient-success",
  danger: "gradient-danger",
  info: "gradient-info",
  warning: "bg-warning",
};

export default function MetricCard({ title, value, subtitle, trend, icon: Icon, variant = "default" }: MetricCardProps) {
  return (
    <div className={cn(
      "glass-card rounded-xl p-5 animate-fade-in",
      "bg-gradient-to-br",
      variantStyles[variant]
    )}>
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
          {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
        </div>
        <div className={cn("flex h-10 w-10 items-center justify-center rounded-lg", iconVariantStyles[variant])}>
          <Icon className="h-5 w-5 text-primary-foreground" />
        </div>
      </div>
      {trend !== undefined && (
        <div className="mt-3 flex items-center gap-1 text-sm">
          {trend > 0 ? (
            <TrendingUp className="h-4 w-4 text-success" />
          ) : trend < 0 ? (
            <TrendingDown className="h-4 w-4 text-destructive" />
          ) : (
            <Minus className="h-4 w-4 text-muted-foreground" />
          )}
          <span className={cn(
            "font-medium",
            trend > 0 ? "text-success" : trend < 0 ? "text-destructive" : "text-muted-foreground"
          )}>
            {trend > 0 ? "+" : ""}{trend}%
          </span>
          <span className="text-muted-foreground">vs last month</span>
        </div>
      )}
    </div>
  );
}
