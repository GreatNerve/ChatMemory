import { cn } from "@/lib/cn";
import { TextareaHTMLAttributes } from "react";

export function BrutalTextarea({
  className,
  ...props
}: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        "w-full min-h-28 border-2 border-[var(--cm-border)] bg-[var(--cm-bg)] px-3 py-2 text-sm text-[var(--cm-text)] font-body outline-none focus:outline-2 focus:outline-offset-2 focus:outline-[var(--cm-accent)] resize-y",
        className,
      )}
      {...props}
    />
  );
}
