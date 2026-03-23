"use client";
import { useState } from "react";
import { Session } from "@/lib/api";
import { History, ChevronUp, ExternalLink, Calendar, Clock } from "lucide-react";

const STATUS_STYLE: Record<string, string> = {
  completed: "bg-emerald-50 text-emerald-700 border-emerald-100",
  pending:   "bg-amber-50  text-amber-700  border-amber-100",
  error:     "bg-rose-50   text-rose-700   border-rose-100",
};

export function SessionHistory({ sessions, onSelect }: { sessions: Session[]; onSelect: (s: Session) => void }) {
  const [open, setOpen] = useState(true);
  if (!sessions.length) return null;
  return (
    <div className="card overflow-hidden anim-in">
      <button onClick={() => setOpen(!open)} className={`w-full flex items-center gap-3 px-6 py-4 transition-all ${open ? "bg-slate-50 border-b border-slate-100" : "bg-white"}`}>
        <div className="w-8 h-8 bg-slate-900 rounded-lg flex items-center justify-center text-white flex-shrink-0"><History size={16} /></div>
        <div className="text-left">
          <p className="text-sm font-black text-slate-800">Patient Diagnostics Archive</p>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest">{sessions.length} cases</p>
        </div>
        <div className={`ml-auto transition-transform ${open ? "" : "rotate-180"}`}><ChevronUp size={16} className="text-slate-400" /></div>
      </button>

      {open && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 text-[10px] font-black text-slate-400 uppercase tracking-widest border-b border-slate-100">
                {["ID","Modality","Diagnosis","Confidence","Status","Date","Report"].map(h => (
                  <th key={h} className={`px-5 py-3 ${h==="Report"?"text-right":"text-left"}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {sessions.map(s => {
                const diag = s.diagnosis_output?.diagnosis_label ?? "—";
                const stat = s.status === "complete" ? "completed" : s.status === "error" ? "error" : "pending";
                const date = s.created_at ? new Date(s.created_at) : null;
                return (
                  <tr key={s.session_id} onClick={() => onSelect(s)} className="group hover:bg-blue-50/40 cursor-pointer transition-all">
                    <td className="px-5 py-4"><span className="font-mono text-[10px] text-slate-400 group-hover:text-blue-600 font-bold">{s.session_id.slice(0,8)}</span></td>
                    <td className="px-5 py-4"><span className="text-xs font-black text-slate-700 uppercase">{s.modality ?? "—"}</span></td>
                    <td className="px-5 py-4"><span className="text-xs font-bold text-slate-800 capitalize">{diag.replace(/_/g," ")}</span></td>
                    <td className="px-5 py-4">
                      {s.confidence_score ? (
                        <div className="flex items-center gap-2">
                          <div className="w-10 h-1.5 bg-slate-100 rounded-full overflow-hidden"><div className="h-full bg-blue-500" style={{ width: `${s.confidence_score}%` }} /></div>
                          <span className="text-[11px] font-black text-slate-900">{Number(s.confidence_score).toFixed(1)}%</span>
                        </div>
                      ) : <span className="text-slate-300">—</span>}
                    </td>
                    <td className="px-5 py-4"><span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full border text-[10px] font-black uppercase ${STATUS_STYLE[stat] ?? STATUS_STYLE.pending}`}>{stat}</span></td>
                    <td className="px-5 py-4">
                      <div className="flex flex-col">
                        <span className="text-[11px] font-bold text-slate-600 flex items-center gap-1"><Calendar size={9} />{date?.toLocaleDateString() ?? "—"}</span>
                        <span className="text-[9px] text-slate-400 flex items-center gap-1"><Clock size={9} />{date?.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}) ?? "—"}</span>
                      </div>
                    </td>
                    <td className="px-5 py-4 text-right">
                      {s.report_link
                        ? <a href={s.report_link} target="_blank" rel="noopener noreferrer" onClick={e=>e.stopPropagation()} className="inline-flex items-center gap-1 text-[10px] font-black text-emerald-600 hover:text-emerald-800">Report <ExternalLink size={11} /></a>
                        : <span className="text-slate-200 text-[10px]">—</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
