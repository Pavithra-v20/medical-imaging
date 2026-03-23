"use client";
import { Session } from "@/lib/api";
import { sessionsApi } from "@/lib/api";
import { Brain, TrendingUp, Shield, AlertCircle, CheckCircle, Activity, Eye } from "lucide-react";

const CLASS_CFG: Record<string, { label: string; bg: string; border: string }> = {
  glioma:     { label: "Glioma Detected",    bg: "from-rose-600 to-rose-800",    border: "border-l-rose-500"    },
  meningioma: { label: "Meningioma Detected", bg: "from-amber-500 to-orange-700", border: "border-l-amber-500"   },
  pituitary:  { label: "Pituitary Tumor",     bg: "from-violet-600 to-indigo-800",border: "border-l-violet-500"  },
  no_tumor:   { label: "No Tumor Detected",   bg: "from-emerald-500 to-teal-700", border: "border-l-emerald-500" },
};

const SEVERITY_ICON: Record<string, React.ReactNode> = {
  Normal:        <CheckCircle size={14} className="text-emerald-400" />,
  Benign:        <AlertCircle size={14} className="text-amber-400"   />,
  Malignant:     <AlertCircle size={14} className="text-rose-400"    />,
  Indeterminate: <Activity    size={14} className="text-slate-400"   />,
};

export function ResultPanel({ session, originalImage }: { session: Session; originalImage: string }) {
  const dout    = session.diagnosis_output ?? {};
  const probs   = dout.class_probabilities ?? {};
  const entries = Object.entries(probs);
  const top     = entries.length > 0 ? entries.reduce((a, b) => b[1] > a[1] ? b : a, ["unknown", 0])[0] : "unknown";
  const cfg     = CLASS_CFG[top] ?? CLASS_CFG.no_tumor;
  const conf    = Number(session.confidence_score ?? 0);
  const sevIcon = SEVERITY_ICON[dout.severity ?? "Indeterminate"] ?? SEVERITY_ICON.Indeterminate;
  // Optional chaining is fine.

  return (
    <div className="space-y-5 anim-up">
      {/* Banner */}
      <div className={`card overflow-hidden border-l-8 ${cfg.border}`}>
        <div className={`px-6 py-5 bg-gradient-to-r ${cfg.bg} text-white`}>
          <div className="flex items-center gap-1.5 mb-2 opacity-70 text-xs font-black uppercase tracking-widest">
            <Brain size={14} /> AI Determination
          </div>
          <div className="flex items-end justify-between gap-4">
            <div>
              <h2 className="text-3xl font-black leading-tight">{cfg.label}</h2>
              <div className="flex items-center gap-2 mt-2">
                <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/20 text-xs font-bold">
                  {sevIcon} {dout.severity ?? "Indeterminate"}
                </span>
                <span className="text-white/60 text-xs">{dout.confidence_label} confidence</span>
              </div>
            </div>
            <div className="text-right flex-shrink-0">
              <p className="text-5xl font-black">{conf.toFixed(0)}%</p>
              <p className="text-white/60 text-[10px] uppercase tracking-widest mt-1">Confidence</p>
            </div>
          </div>
        </div>
        <div className="h-1.5 bg-slate-100">
          <div className={`h-full bg-gradient-to-r ${cfg.bg} transition-all duration-1000`} style={{ width: `${conf}%` }} />
        </div>
        <div className="px-6 py-3 bg-slate-50 text-xs text-slate-400 flex items-center gap-2">
          <Shield size={12} /> {session.model_used ?? "ViT-L16-fe + Xception"}
          <span className="ml-auto font-mono">{session.session_id.slice(0, 12)}…</span>
        </div>
      </div>

      {/* Scan Display */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-4">
          <Eye size={16} className="text-indigo-600" />
          <h3 className="font-bold text-slate-900">Analyzed Scan</h3>
        </div>
        <div className="flex justify-center">
          <div className="w-full md:w-1/2">
            <Frame label="Original Scan" sub="Pre-processed input">
              {originalImage
                ? <img src={originalImage} alt="original" className="w-full max-h-[300px] object-contain rounded-xl" />
                : <p className="text-slate-500 text-xs text-center">No image</p>}
            </Frame>
          </div>
        </div>
      </div>

      {/* Probs + Rec */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp size={15} className="text-blue-600" />
            <h3 className="font-bold text-slate-900 text-sm">Class Distribution</h3>
          </div>
          <div className="space-y-3">
            {entries.sort(([,a],[,b]) => b-a).map(([cls, p]) => (
              <div key={cls}>
                <div className="flex justify-between mb-1">
                  <span className={`text-xs font-bold capitalize ${cls === top ? "text-slate-900" : "text-slate-400"}`}>{cls.replace(/_/g," ")}</span>
                  <span className={`text-xs font-black ${cls === top ? "text-blue-600" : "text-slate-400"}`}>{p.toFixed(2)}%</span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full transition-all duration-700 ${cls === top ? "bg-blue-600" : "bg-slate-300"}`} style={{ width: `${p}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card p-5 border-t-4 border-t-amber-400">
          <div className="flex items-center gap-2 mb-3">
            <AlertCircle size={15} className="text-amber-500" />
            <h3 className="font-bold text-slate-900 text-sm">Clinical Recommendation</h3>
          </div>
          <p className="text-sm text-slate-700 italic bg-amber-50 border border-amber-100 rounded-xl p-3 leading-relaxed">
            &ldquo;{dout.recommended_action ?? "Standard follow-up advised. Refer for radiologist confirmation."}&rdquo;
          </p>
          {(dout.differential_diagnoses ?? []).length > 0 && (
            <div className="mt-4">
              <p className="section-title mb-2">Differentials</p>
              <div className="flex flex-wrap gap-1.5">
                {(dout.differential_diagnoses ?? []).map((d, i) => (
                  <span key={i} className="badge bg-slate-100 text-slate-600 border border-slate-200">{d}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Frame({ label, sub, children, dark }: { label: string; sub: string; children: React.ReactNode; dark?: boolean }) {
  return (
    <div className="space-y-2">
      <div className={`rounded-2xl overflow-hidden flex items-center justify-center min-h-[200px] p-2 ${dark ? "bg-slate-900 ring-2 ring-indigo-500/20" : "bg-slate-900"}`}>
        {children}
      </div>
      <div className="text-center">
        <p className="text-xs font-bold text-slate-700">{label}</p>
        <p className="text-[10px] text-slate-400">{sub}</p>
      </div>
    </div>
  );
}
