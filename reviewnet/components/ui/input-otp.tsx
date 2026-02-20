// InputOTP component - placeholder
// Install input-otp if needed: npm install input-otp

"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

const InputOTP = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement> & { containerClassName?: string }>(
  ({ className, containerClassName, ...props }, ref) => (
    <div className={cn("flex items-center gap-2", containerClassName)}>
      <input
        ref={ref}
        className={cn("disabled:cursor-not-allowed", className)}
        {...props}
      />
    </div>
  )
);
InputOTP.displayName = "InputOTP";

const InputOTPGroup = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex items-center", className)} {...props} />
  )
);
InputOTPGroup.displayName = "InputOTPGroup";

const InputOTPSlot = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement> & { index: number }>(
  ({ className, index, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "relative flex h-10 w-10 items-center justify-center border-y border-r border-input text-sm transition-all first:rounded-l-md first:border-l last:rounded-r-md",
        className
      )}
      {...props}
    />
  )
);
InputOTPSlot.displayName = "InputOTPSlot";

const InputOTPSeparator = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ ...props }, ref) => (
    <div ref={ref} role="separator" {...props}>
      <div className="h-1 w-px bg-border" />
    </div>
  )
);
InputOTPSeparator.displayName = "InputOTPSeparator";

export { InputOTP, InputOTPGroup, InputOTPSlot, InputOTPSeparator };
