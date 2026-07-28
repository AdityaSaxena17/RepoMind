import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import type { TimelineStep } from "../lib/prototype-data";

export function Timeline({ steps }: { steps: TimelineStep[] }) {
  return (
    <div className="space-y-3">
      {steps.map((step) => {
        const Icon =
          step.status === "complete"
            ? CheckCircle2
            : step.status === "active"
              ? Loader2
              : Circle;

        return (
          <div key={step.label} className="flex gap-3">
            <Icon
              className={`mt-0.5 h-4 w-4 shrink-0 ${
                step.status === "complete"
                  ? "text-emerald-300"
                  : step.status === "active"
                    ? "animate-spin text-sky-300"
                    : "text-zinc-600"
              }`}
            />
            <div>
              <p className="text-sm font-medium text-zinc-100">{step.label}</p>
              {step.detail ? (
                <p className="mt-1 text-xs leading-5 text-zinc-500">
                  {step.detail}
                </p>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}
