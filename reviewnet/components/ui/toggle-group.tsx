// ToggleGroup component - placeholder
// Install @radix-ui/react-toggle-group if needed: npm install @radix-ui/react-toggle-group

"use client";

import * as React from "react";
import { type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { toggleVariants } from "@/components/ui/toggle";

const ToggleGroupContext = React.createContext<VariantProps<typeof toggleVariants>>({
  size: "default",
  variant: "default",
});

const ToggleGroup = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof toggleVariants>>(
  ({ className, variant, size, children, ...props }, ref) => (
    <div ref={ref} className={cn("flex items-center justify-center gap-1", className)} {...props}>
      <ToggleGroupContext.Provider value={{ variant, size }}>{children}</ToggleGroupContext.Provider>
    </div>
  )
);

ToggleGroup.displayName = "ToggleGroup";

const ToggleGroupItem = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof toggleVariants>>(
  ({ className, variant, size, ...props }, ref) => {
    const context = React.useContext(ToggleGroupContext);

    return (
      <button
        ref={ref}
        className={cn(
          toggleVariants({
            variant: context.variant || variant,
            size: context.size || size,
          }),
          className
        )}
        {...props}
      />
    );
  }
);

ToggleGroupItem.displayName = "ToggleGroupItem";

export { ToggleGroup, ToggleGroupItem };
