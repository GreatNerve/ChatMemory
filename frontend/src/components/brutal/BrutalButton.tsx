import { cn } from "@/lib/cn";
import { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "ghost" | "danger";

interface BrutalButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

const variants: Record<Variant, string> = {
  primary:
    "bg-[var(--cm-accent)] text-[var(--cm-accent-fg)] border-2 border-[var(--cm-border)] cm-shadow-brutal hover:opacity-90",
  ghost:
    "bg-transparent text-[var(--cm-text)] border-2 border-[var(--cm-border)] hover:bg-[var(--cm-surface-raised)]",
  danger:
    "bg-transparent text-[var(--cm-error)] border-2 border-[var(--cm-error)] hover:bg-[var(--cm-surface-raised)]",
};

export function BrutalButton({
  className,
  variant = "primary",
  disabled,
  children,
  ...props
}: BrutalButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      className={cn(
        "font-mono text-xs uppercase tracking-widest px-4 py-3 min-h-11 transition-transform disabled:opacity-40 disabled:cursor-not-allowed",
        variants[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
