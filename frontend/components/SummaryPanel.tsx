"use client";
import { Session } from "@/lib/api";
import { Microscope, MessageSquare, ChevronRight } from "lucide-react";

export function SummaryPanel({ session }: { session: Session }) {
  const dout = session.diagnosis_output ?? {};
  const note = dout.clinical_note ?? session.clinical_notes ?? "";
  const exp  = session.explanation ?? "";

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="font-black text-slate-900 text-xl">Clinical Intelligence</h2>
        <div className="h-px flex-1 bg-slate-200" />
        <span className="section-title">AI Generated</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="card overflow-hidden">
          <div className="px-5 py-3 bg-blue-50 border-b border-blue-100 flex items-center gap-2">
            <Microscope size={15} className="text-blue-600" />
            <h3 className="font-bold text-slate-900 text-sm">Clinical Summary</h3>
            <span className="ml-auto text-[10px] font-black text-blue-400 uppercase tracking-widest">Agent 1</span>
          </div>
          <div className="p-5">
            <p className="text-sm text-slate-600 leading-relaxed">{note || "No clinical notes generated."}</p>
            {(dout.differential_diagnoses ?? []).length > 0 && (
              <div className="mt-4 pt-4 border-t border-slate-100">
                <p className="section-title mb-2">Differential Diagnoses</p>
                <div className="flex flex-wrap gap-1.5">
                  {(dout.differential_diagnoses ?? []).map((d, i) => (
                    <span key={i} className="inline-flex items-center gap-1 px-2.5 py-1 bg-slate-50 border border-slate-200 rounded-lg text-xs font-semibold text-slate-700">
                      <ChevronRight size={10} className="text-blue-500" />{d}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="card overflow-hidden">
          <div className="px-5 py-3 bg-indigo-50 border-b border-indigo-100 flex items-center gap-2">
            <MessageSquare size={15} className="text-indigo-600" />
            <h3 className="font-bold text-slate-900 text-sm">Visual Findings</h3>
            <span className="ml-auto text-[10px] font-black text-indigo-400 uppercase tracking-widest">Agent 2</span>
          </div>
          <div className="p-5 max-h-72 overflow-y-auto">
            {exp ? (
              <div className="space-y-2">
                {exp.split("\n").filter(l => l.trim()).map((line, i) => {
                  const hdr = /^\d+\./.test(line.trim()) || line.trim().endsWith(":");
                  return <p key={i} className={`text-sm leading-relaxed ${hdr ? "font-bold text-slate-900 border-l-2 border-indigo-400 pl-3" : "text-slate-600 pl-3"}`}>{line.trim()}</p>;
                })}
              </div>
            ) : <p className="text-sm text-slate-400 italic">Visual explanation unavailable.</p>}
          </div>
        </div>
      </div>
    </div>
  );
}
