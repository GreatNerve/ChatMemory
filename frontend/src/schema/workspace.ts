import { z } from "zod";

export const createWorkspaceSchema = z.object({
  name: z.string().min(1, "Name required").max(120),
  file: z
    .custom<File>((v) => v instanceof File, "WhatsApp .txt file required")
    .refine((f) => f.name.toLowerCase().endsWith(".txt"), "Must be a .txt file"),
});

export const askSchema = z.object({
  question: z.string().min(3, "Question too short"),
  speaker: z.string().optional(),
  dateFrom: z.string().optional(),
  dateTo: z.string().optional(),
});

export const trainPersonaSchema = z.object({
  consent: z.literal(true, { message: "Consent required" }),
  forceThin: z.boolean().optional(),
  forceRetrain: z.boolean().optional(),
});
