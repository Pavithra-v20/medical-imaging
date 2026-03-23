"use client";
import { CheckCircle2, Loader2, AlertCircle, Circle } from "lucide-react";

const STEPS = [
  { num: 1, label: "Clinical Diagnosis",   sub: "ResNet18 Inference",    color: "text-blue-600"   },
  { num: 2, label: "Report Generation",   sub: "Clinical PDF Report",      color: "text-violet-600" },
];

export function PipelineProgress({ currentStep }: { currentStep: number }) {
  const done  = currentStep === 3;
  const error = currentStep === -1;
  const pct   = done ? 100 : error ? 100 : Math.max(0, (currentStep - 1) / 2) * 100;

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="font-black text-slate-900 text-sm">Pipeline Orchestrator</p>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-medium mt-0.5">LangGraph · 2 Agents</p>
        </div>
        <span className={`badge border text-[10px] py-1 ${done ? "bg-emerald-50 text-emerald-700 border-emerald-100" : error ? "bg-rose-50 text-rose-700 border-rose-100" : "bg-blue-50 text-blue-700 border-blue-100"}`}>
          {done ? "✓ Complete" : error ? "✗ Error" : "⏳ Running"}
        </span>
      </div>

      <div className="h-1.5 bg-slate-100 rounded-full mb-5 overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-700 ${error ? "bg-rose-500" : "bg-gradient-to-r from-blue-500 to-indigo-600"}`} style={{ width: `${pct}%` }} />
      </div>

      <div className="space-y-3">
        {STEPS.map(step => {
          const isDone    = currentStep > step.num || done;
          const isRunning = currentStep === step.num;
          const isError   = error;
          const isPending = !isDone && !isRunning && !isError;
          return (
            <div key={step.num} className={`flex items-center gap-3 transition-all ${isPending ? "opacity-40" : ""}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 border-2 transition-all ${
                isDone    ? "bg-emerald-500 border-emerald-500 text-white" :
                isRunning ? "border-blue-400 bg-blue-50 text-blue-600 animate-pulse" :
                isError   ? "bg-rose-50 border-rose-400 text-rose-500" :
                            "bg-white border-slate-200 text-slate-300"
              }`}>
                {isDone    ? <CheckCircle2 size={15} /> :
                 isRunning ? <Loader2 size={15} className="animate-spin" /> :
                 isError   ? <AlertCircle size={15} /> :
                             <Circle size={15} strokeWidth={1.5} />}
              </div>
              <div className="flex-1 min-w-0">
                <p className={`text-xs font-bold ${isDone ? "text-emerald-600" : isRunning ? step.color : "text-slate-500"}`}>
                  {step.label}
                  {isRunning && <span className="ml-2 text-[9px] text-blue-500 font-black uppercase tracking-widest">● Active</span>}
                </p>
                <p className="text-[10px] text-slate-400 truncate">{step.sub}</p>
              </div>
              <span className="text-[10px] font-black text-slate-300">0{step.num}</span>
            </div>
          );
        })}
      </div>

      {done  && <div className="mt-4 p-3 bg-emerald-50 border border-emerald-100 rounded-xl text-xs text-emerald-700 font-bold flex items-center gap-2 justify-center"><CheckCircle2 size={13} />All agents completed</div>}
      {error && <div className="mt-4 p-3 bg-rose-50 border border-rose-100 rounded-xl text-xs text-rose-700 font-bold flex items-center gap-2 justify-center"><AlertCircle size={13} />Pipeline error — check logs</div>}
    </div>
  );
}
