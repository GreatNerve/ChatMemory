import { cn } from "@/lib/cn";
import { forwardRef, InputHTMLAttributes } from "react";

export const BrutalInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function BrutalInput({ className, ...props }, ref) {
    return (
      <input
        ref={ref}
        className={cn(
          "w-full border-2 border-[var(--cm-border)] bg-[var(--cm-bg)] px-3 py-2 text-sm text-[var(--cm-text)] font-body outline-none focus:outline-2 focus:outline-offset-2 focus:outline-[var(--cm-accent)]",
          className,
        )}
        {...props}
      />
    );
  },
);
