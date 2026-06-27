import { cn } from "@/lib/cn";
import { HTMLAttributes } from "react";

export function BrutalPanel({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "border-2 border-[var(--cm-border)] bg-[var(--cm-surface)] p-4",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
