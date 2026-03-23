"use client";
import { useEffect, useState, useCallback } from "react";
import { Brain, Activity, Users, Stethoscope, BarChart2, RefreshCw, Plus, X, AlertCircle, Cpu, Clock, ShieldCheck, CheckCircle2, Database } from "lucide-react";
import { UploadForm }       from "@/components/UploadForm";
import { ResultPanel }      from "@/components/ResultPanel";
import { SummaryPanel }     from "@/components/SummaryPanel";
import { PipelineProgress } from "@/components/PipelineProgress";
import { SessionHistory }   from "@/components/SessionHistory";
import { patientsApi, physiciansApi, techniciansApi, sessionsApi, Patient, Physician, Technician, Session } from "@/lib/api";

export default function Home() {
  const [patients,    setPatients]    = useState<Patient[]>([]);
  const [physicians,  setPhysicians]  = useState<Physician[]>([]);
  const [technicians, setTechnicians] = useState<Technician[]>([]);
  const [sessions,    setSessions]    = useState<Session[]>([]);
  const [current,     setCurrent]     = useState<Session | null>(null);
  const [origImg,     setOrigImg]     = useState("");
  const [loading,     setLoading]     = useState(false);
  const [step,        setStep]        = useState(0);
  const [error,       setError]       = useState("");
  const [view,        setView]        = useState<"dash"|"result">("dash");
  const [backendOk,   setBackendOk]   = useState<boolean|null>(null);

  // Modals
  const [addPat, setAddPat] = useState(false);
  const [addPhy, setAddPhy] = useState(false);
  const [addTec, setAddTec] = useState(false);
  const [newPat, setNewPat] = useState({ name:"", dob:"", gender:"Male", contact:"" });
  const [newPhy, setNewPhy] = useState({ name:"", specialization:"", email:"" });
  const [newTec, setNewTec] = useState({ name:"", email:"", password:"", role:"technician" });

  const load = useCallback(async () => {
    try {
      const [pa, ph, te, se] = await Promise.all([patientsApi.list(), physiciansApi.list(), techniciansApi.list(), sessionsApi.list()]);
      setPatients(pa); setPhysicians(ph); setTechnicians(te); setSessions(se);
      setBackendOk(true);
    } catch { setBackendOk(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function analyze(pid: string, tid: string, phid: string, mod: string, file: File, img: string) {
    setLoading(true); setError(""); setStep(1); setOrigImg(img); setView("result");
    try {
      setStep(1); await new Promise(r => setTimeout(r, 500));
      const s = await sessionsApi.run(pid, tid, phid, mod, file);
      setStep(2); await new Promise(r => setTimeout(r, 300));
      setStep(3);
      setCurrent(s); setSessions(prev => [s, ...prev]);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Pipeline failed");
      setStep(-1);
    } finally { setLoading(false); }
  }

  async function savePat(e: React.FormEvent) {
    e.preventDefault();
    try { await patientsApi.create(newPat as any); setAddPat(false); setNewPat({ name:"", dob:"", gender:"Male", contact:"" }); load(); }
    catch (e: any) { setError(e.message); }
  }
  async function savePhy(e: React.FormEvent) {
    e.preventDefault();
    try { await physiciansApi.create(newPhy as any); setAddPhy(false); setNewPhy({ name:"", specialization:"", email:"" }); load(); }
    catch (e: any) { setError(e.message); }
  }
  async function saveTec(e: React.FormEvent) {
    e.preventDefault();
    try { await techniciansApi.register(newTec); setAddTec(false); setNewTec({ name:"", email:"", password:"", role:"technician" }); load(); }
    catch (e: any) { setError(e.message); }
  }

  const done = sessions.filter(s => s.status === "complete");
  const avgConf = done.length > 0 ? done.reduce((a, s) => a + Number(s.confidence_score ?? 0), 0) / done.length : 0;

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col">
      {/* ── NAVBAR ── */}
      <header className="bg-[#0f172a] sticky top-0 z-50 border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-900/40">
              <Brain size={20} className="text-white" />
            </div>
            <div>
              <p className="text-white font-black text-base leading-none">MedAI Diagnostics</p>
              <p className="text-slate-400 text-[9px] font-medium uppercase tracking-widest">Explainable AI Radiology System</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {backendOk !== null && (
              <span className={`hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-[10px] font-bold ${backendOk ? "bg-emerald-900/30 border-emerald-700 text-emerald-400" : "bg-rose-900/30 border-rose-700 text-rose-400"}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${backendOk ? "bg-emerald-400 anim-pulse" : "bg-rose-400"}`} />
                {backendOk ? "System Online" : "Offline"}
              </span>
            )}
            <button onClick={load} className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/10 transition-all"><RefreshCw size={15} /></button>
            {view === "result" && (
              <button onClick={() => { setView("dash"); setCurrent(null); setStep(0); setError(""); }} className="btn-primary text-sm py-2">
                <Plus size={15} /> New Scan
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8 space-y-7">
        {/* KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 anim-up">
          {[
            { icon:<Users size={20}/>,      label:"Patients",     val:patients.length,                      col:"bg-blue-50 text-blue-600"    },
            { icon:<Stethoscope size={20}/>, label:"Physicians",   val:physicians.length,                    col:"bg-indigo-50 text-indigo-600" },
            { icon:<Activity size={20}/>,   label:"Total Cases",  val:sessions.length,                      col:"bg-violet-50 text-violet-600" },
            { icon:<BarChart2 size={20}/>,  label:"Avg Confidence",val:avgConf>0?`${avgConf.toFixed(1)}%`:"—",col:"bg-emerald-50 text-emerald-600"},
          ].map(k => (
            <div key={k.label} className="kpi-card card-hover">
              <div className={`w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 ${k.col}`}>{k.icon}</div>
              <div>
                <p className="text-xl font-black text-slate-900 leading-none">{k.val}</p>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mt-0.5">{k.label}</p>
              </div>
            </div>
          ))}
        </div>

        {/* ── DASHBOARD ── */}
        {view === "dash" && (
          <div className="grid grid-cols-12 gap-6 anim-up delay-1">
            <div className="col-span-12 lg:col-span-8 space-y-5">
              {/* Hero */}
              <div className="card overflow-hidden" style={{ background: "linear-gradient(135deg,#0f172a 0%,#1e3a8a 55%,#1d4ed8 100%)" }}>
                <div className="p-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
                  <div>
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-500/20 border border-blue-500/30 text-blue-300 text-[10px] font-bold mb-4">
                      <Cpu size={11} /> ResNet18 · LangGraph
                    </span>
                    <h1 className="text-3xl md:text-4xl font-black text-white leading-tight mb-2">
                      AI-Powered Brain MRI<br />Diagnostics
                    </h1>
                    <p className="text-blue-200 text-sm max-w-md leading-relaxed">
                      Upload a brain MRI scan (JPEG, PNG, DICOM). Our AI pipeline delivers rapid diagnoses and clinical reports.
                    </p>
                  </div>
                  <div className="flex gap-3 flex-shrink-0">
                    {[
                      { icon:<ShieldCheck size={20} className="text-emerald-300"/>, val:done.length, label:"Completed" },
                      { icon:<CheckCircle2 size={20} className="text-blue-300"/>,   val:"99%",       label:"Accuracy"  },
                    ].map(s => (
                      <div key={s.label} className="flex flex-col items-center p-4 rounded-2xl bg-white/10 border border-white/20 text-center min-w-[80px]">
                        {s.icon}
                        <p className="text-white font-black text-lg mt-1 leading-none">{s.val}</p>
                        <p className="text-blue-200 text-[9px] uppercase tracking-widest font-bold mt-0.5">{s.label}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Error */}
              {error && (
                <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-start gap-3 anim-in">
                  <AlertCircle className="text-rose-500 mt-0.5 flex-shrink-0" size={17} />
                  <div className="flex-1"><p className="text-sm font-bold text-rose-700">Pipeline Error</p><p className="text-xs text-rose-600 mt-0.5">{error}</p></div>
                  <button onClick={() => setError("")} className="text-rose-400 hover:text-rose-600"><X size={14} /></button>
                </div>
              )}

              {/* Upload Card */}
              <div className="card">
                <div className="px-6 py-4 border-b border-slate-100 flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center"><Activity size={16} className="text-white" /></div>
                  <div>
                    <h2 className="font-bold text-slate-900 text-sm">New Diagnostic Session</h2>
                    <p className="text-[11px] text-slate-400">Fill in patient details and upload the scan</p>
                  </div>
                </div>
                <div className="p-6">
                  <UploadForm patients={patients} physicians={physicians} technicians={technicians} onSubmit={analyze} isLoading={loading} />
                </div>
              </div>
            </div>

            <div className="col-span-12 lg:col-span-4 space-y-5">
              {step > 0 && <PipelineProgress currentStep={step} />}

              {/* Quick Add */}
              <div className="card p-5">
                <p className="section-title mb-3">Quick Add</p>
                <div className="space-y-2">
                  {[
                    { label:"Add Patient",    icon:<Users size={14}/>,       fn:()=>setAddPat(true), cls:"text-blue-600 bg-blue-50 hover:bg-blue-100 border-blue-100" },
                    { label:"Add Physician",  icon:<Stethoscope size={14}/>, fn:()=>setAddPhy(true), cls:"text-indigo-600 bg-indigo-50 hover:bg-indigo-100 border-indigo-100" },
                    { label:"Add Technician", icon:<Database size={14}/>,    fn:()=>setAddTec(true), cls:"text-violet-600 bg-violet-50 hover:bg-violet-100 border-violet-100" },
                  ].map(b => (
                    <button key={b.label} onClick={b.fn} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl border font-semibold text-sm transition-all ${b.cls}`}>
                      {b.icon}{b.label}<span className="ml-auto text-xs opacity-40">→</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Model Spec */}
              <div className="card overflow-hidden">
                <div className="px-5 py-3 border-b border-slate-100 flex items-center gap-2">
                  <Cpu size={14} className="text-slate-400" /><p className="section-title">Model Specification</p>
                </div>
                <div className="p-5 space-y-2">
                  {[
                    ["Architecture","ViT-L16-fe + Xception"],
                    ["Classes","4 (Glioma, Meningioma…)"],
                    ["Explainability","Grad-CAM (Xception)"],
                    ["Clinical LLM","Gemini 2.0 Flash"],
                    ["Orchestration","LangGraph (3 Agents)"],
                    ["Report","PDF + Google Drive"],
                  ].map(([k,v]) => (
                    <div key={k} className="flex justify-between py-1.5 border-b border-slate-50 last:border-0">
                      <span className="text-xs text-slate-400 font-medium">{k}</span>
                      <span className="text-xs font-bold text-slate-700 text-right max-w-[55%] truncate">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── RESULT ── */}
        {view === "result" && current && (
          <div className="grid grid-cols-12 gap-6 anim-up">
            <div className="col-span-12 lg:col-span-8 space-y-5">
              <ResultPanel session={current} originalImage={origImg} />
              <SummaryPanel session={current} />
            </div>
            <div className="col-span-12 lg:col-span-4 space-y-5">
              <PipelineProgress currentStep={step} />
              <div className="card p-5">
                <p className="section-title mb-3">Session Metadata</p>
                <div className="space-y-2">
                  {[
                    ["Session ID", current.session_id?.slice(0,16)+"…"],
                    ["Modality",   (current.modality ?? "—").toUpperCase()],
                    ["Status",     current.status],
                    ["Created",    new Date(current.created_at!).toLocaleString()],
                  ].map(([k,v]) => (
                    <div key={k} className="flex justify-between py-1.5 border-b border-slate-50 last:border-0">
                      <span className="text-xs text-slate-400 font-medium">{k}</span>
                      <span className="text-xs font-bold text-slate-700 font-mono">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
              {current.report_link && (
                <a href={current.report_link} target="_blank" rel="noopener noreferrer"
                  className="btn-primary w-full justify-center py-3 text-sm">
                  View Full Report →
                </a>
              )}
            </div>
          </div>
        )}

        {/* History */}
        {sessions.length > 0 && (
          <div className="anim-up delay-2">
            <SessionHistory sessions={sessions} onSelect={s => { setCurrent(s); setView("result"); setStep(4); window.scrollTo({top:0,behavior:"smooth"}); }} />
          </div>
        )}
      </main>

      {/* FOOTER */}
      <footer className="border-t border-slate-200 bg-white mt-8 py-5">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-2"><Brain size={15} className="text-blue-600"/><span className="text-sm font-bold text-slate-600">MedAI Diagnostics v1.0</span></div>
          <p className="text-xs text-slate-400 hidden md:block">For research use · Consult a qualified radiologist</p>
          <div className="flex items-center gap-1 text-xs text-slate-400"><Clock size={11}/>{new Date().toLocaleDateString("en-IN",{dateStyle:"medium"})}</div>
        </div>
      </footer>

      {/* MODALS */}
      {addPat && (
        <Modal title="Register Patient" onClose={() => setAddPat(false)}>
          <form onSubmit={savePat} className="space-y-4">
            <F label="Full Name" value={newPat.name} set={v=>setNewPat(p=>({...p,name:v}))} placeholder="Rahul Sharma" />
            <div className="grid grid-cols-2 gap-3">
              <F label="Date of Birth" type="date" value={newPat.dob} set={v=>setNewPat(p=>({...p,dob:v}))} />
              <div><p className="label">Gender</p>
                <select value={newPat.gender} onChange={e=>setNewPat(p=>({...p,gender:e.target.value}))} className="input">
                  <option>Male</option><option>Female</option><option>Other</option>
                </select>
              </div>
            </div>
            <F label="Contact" value={newPat.contact} set={v=>setNewPat(p=>({...p,contact:v}))} placeholder="Phone or email" />
            <button type="submit" className="btn-primary w-full justify-center py-3">Register Patient</button>
          </form>
        </Modal>
      )}
      {addPhy && (
        <Modal title="Add Physician" onClose={() => setAddPhy(false)}>
          <form onSubmit={savePhy} className="space-y-4">
            <F label="Full Name"      value={newPhy.name}           set={v=>setNewPhy(p=>({...p,name:v}))}           placeholder="Dr. Priya Nair" />
            <F label="Specialization" value={newPhy.specialization} set={v=>setNewPhy(p=>({...p,specialization:v}))} placeholder="Radiology" />
            <F label="Email" type="email" value={newPhy.email}      set={v=>setNewPhy(p=>({...p,email:v}))}          placeholder="doctor@hospital.com" />
            <button type="submit" className="btn-primary w-full justify-center py-3">Add Physician</button>
          </form>
        </Modal>
      )}
      {addTec && (
        <Modal title="Register Technician" onClose={() => setAddTec(false)}>
          <form onSubmit={saveTec} className="space-y-4">
            <F label="Full Name" value={newTec.name}     set={v=>setNewTec(t=>({...t,name:v}))}     placeholder="Tech Name" />
            <F label="Email" type="email" value={newTec.email}    set={v=>setNewTec(t=>({...t,email:v}))}    placeholder="tech@lab.com" />
            <F label="Password" type="password" value={newTec.password} set={v=>setNewTec(t=>({...t,password:v}))} placeholder="••••••••" />
            <button type="submit" className="btn-primary w-full justify-center py-3">Register Technician</button>
          </form>
        </Modal>
      )}
    </div>
  );
}

function Modal({ title, onClose, children }: { title:string; onClose:()=>void; children:React.ReactNode }) {
  return (
    <div className="fixed inset-0 bg-slate-900/70 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
      <div className="card w-full max-w-md shadow-2xl anim-up">
        <div className="flex items-center gap-3 px-6 py-4 border-b border-slate-100 bg-slate-50">
          <h3 className="font-black text-slate-900">{title}</h3>
          <button onClick={onClose} className="ml-auto p-1.5 hover:bg-slate-200 rounded-lg text-slate-400 transition-colors"><X size={16} /></button>
        </div>
        <div className="p-6">{children}</div>
      </div>
    </div>
  );
}

function F({ label, value, set, type="text", placeholder="" }: { label:string; value:string; set:(v:string)=>void; type?:string; placeholder?:string }) {
  return (
    <div>
      <p className="label">{label}</p>
      <input type={type} value={value} onChange={e=>set(e.target.value)} placeholder={placeholder} className="input" required />
    </div>
  );
}
