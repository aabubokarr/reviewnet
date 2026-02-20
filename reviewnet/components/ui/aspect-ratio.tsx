// AspectRatio component - placeholder
// Install @radix-ui/react-aspect-ratio if needed: npm install @radix-ui/react-aspect-ratio

import * as React from "react";

interface AspectRatioProps extends React.HTMLAttributes<HTMLDivElement> {
  ratio?: number;
  children: React.ReactNode;
}

const AspectRatio = React.forwardRef<HTMLDivElement, AspectRatioProps>(
  ({ ratio = 16 / 9, children, className, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={className}
        style={{ aspectRatio: ratio.toString(), ...props.style }}
        {...props}
      >
        {children}
      </div>
    );
  }
);

AspectRatio.displayName = "AspectRatio";

export { AspectRatio };
