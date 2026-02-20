// HoverCard component - placeholder
// Install @radix-ui/react-hover-card if needed: npm install @radix-ui/react-hover-card

"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

const HoverCard = ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div {...props}>{children}</div>
);

const HoverCardTrigger = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn(className)} {...props} />
  )
);
HoverCardTrigger.displayName = "HoverCardTrigger";

const HoverCardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement> & { align?: "start" | "center" | "end"; sideOffset?: number }>(
  ({ className, align = "center", sideOffset = 4, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "z-50 w-64 rounded-md border bg-popover p-4 text-popover-foreground shadow-md outline-none",
        className
      )}
      {...props}
    />
  )
);
HoverCardContent.displayName = "HoverCardContent";

export { HoverCard, HoverCardTrigger, HoverCardContent };
