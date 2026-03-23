"use client";
import { useState, useRef } from "react";
import { Upload, User, Stethoscope, ImageIcon, ChevronDown, X, Zap, Brain } from "lucide-react";
import { Patient, Physician, Technician } from "@/lib/api";

interface Props {
  patients: Patient[]; physicians: Physician[]; technicians: Technician[];
  onSubmit: (pid: string, tid: string, phid: string, mod: string, file: File, preview: string) => void;
  isLoading: boolean;
}

const MODALITIES = [
  { id: "mri",  label: "MRI",   sub: "Magnetic Resonance" },
];

export function UploadForm({ patients, physicians, technicians, onSubmit, isLoading }: Props) {
  const [patientId,    setPatientId]    = useState("");
  const [technicianId, setTechnicianId] = useState("");
  const [physicianId,  setPhysicianId]  = useState("");
  const [modality,     setModality]     = useState("mri");
  const [file,         setFile]         = useState<File | null>(null);
  const [preview,      setPreview]      = useState("");
  const [drag,         setDrag]         = useState(false);
  const ref = useRef<HTMLInputElement>(null);

  function pick(f: File) {
    if (!f.type.startsWith("image/") && !f.name.toLowerCase().endsWith(".dcm") && !f.name.toLowerCase().endsWith(".dicom")) return;
    setFile(f);
    const r = new FileReader();
    r.onload = e => setPreview(e.target?.result as string);
    r.readAsDataURL(f);
  }

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (patientId && technicianId && physicianId && file && modality)
      onSubmit(patientId, technicianId, physicianId, modality, file, preview);
  }

  const valid = patientId && technicianId && physicianId && file && modality;

  return (
    <form onSubmit={submit} className="space-y-6">
      {/* Selects */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Sel label="Patient"    icon={<User size={13} className="text-blue-500" />}    value={patientId}    onChange={setPatientId}    placeholder="Select patient…"    options={patients.map(p   => ({ value: String(p.patient_id),   label: p.name }))} />
        <Sel label="Technician" icon={<Stethoscope size={13} className="text-violet-500" />} value={technicianId} onChange={setTechnicianId} placeholder="Select technician…" options={technicians.map(t => ({ value: String(t.technician_id), label: t.name }))} />
        <Sel label="Physician"  icon={<Stethoscope size={13} className="text-indigo-500" />} value={physicianId}  onChange={setPhysicianId}  placeholder="Select physician…"  options={physicians.map(p  => ({ value: String(p.physician_id),  label: `Dr. ${p.name}` }))} />
      </div>

      {/* Modality */}
      <div>
        <p className="label">Imaging Modality</p>
        <div className="flex gap-3">
          {MODALITIES.map(m => (
            <button key={m.id} type="button" onClick={() => setModality(m.id)}
              className={`flex-1 py-3 rounded-xl border-2 font-bold text-sm flex flex-col items-center gap-0.5 transition-all ${
                modality === m.id ? "border-blue-600 bg-blue-600 text-white" : "border-slate-200 bg-white text-slate-500 hover:border-slate-300"
              }`}>
              {m.label}
              <span className={`text-[9px] font-medium uppercase tracking-widest ${modality === m.id ? "text-blue-200" : "text-slate-400"}`}>{m.sub}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Drop zone */}
      <div>
        <p className="label flex items-center gap-1.5"><ImageIcon size={12} /> Scan Image</p>
        <div
          onDragOver={e => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) pick(f); }}
          onClick={() => ref.current?.click()}
          className={`relative cursor-pointer rounded-2xl border-2 border-dashed flex flex-col items-center justify-center min-h-[180px] transition-all ${
            drag ? "border-blue-500 bg-blue-50" : preview ? "border-emerald-300 bg-emerald-50/30" : "border-slate-200 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/20"
          }`}>
          {preview ? (
            <div className="w-full p-4 flex flex-col items-center gap-3">
              <img src={preview} alt="Preview" className="max-h-36 object-contain rounded-xl shadow border-4 border-white" />
              <p className="text-xs font-bold text-slate-700 truncate max-w-xs">{file?.name}</p>
              <button type="button" onClick={e => { e.stopPropagation(); setFile(null); setPreview(""); }}
                className="absolute top-2 right-2 p-1.5 bg-rose-500 text-white rounded-full hover:bg-rose-600">
                <X size={12} />
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 py-8 text-center px-6">
              <div className="w-12 h-12 rounded-2xl bg-white shadow border border-slate-100 flex items-center justify-center">
                <Upload size={22} className="text-blue-500" />
              </div>
              <div>
                <p className="text-sm font-bold text-slate-700">Drop scan or <span className="text-blue-600 underline">browse</span></p>
                <p className="text-xs text-slate-400 mt-0.5">JPEG, PNG, DICOM (.dcm) · Brain MRI</p>
              </div>
            </div>
          )}
        </div>
        <input ref={ref} type="file" accept="image/*,.dcm,.dicom" className="hidden" onChange={e => e.target.files?.[0] && pick(e.target.files[0])} />
      </div>

      {/* Submit */}
      <button type="submit" disabled={!valid || isLoading}
        className={`w-full py-4 rounded-xl font-black text-base flex items-center justify-center gap-3 transition-all ${
          valid && !isLoading ? "bg-slate-900 hover:bg-blue-700 text-white shadow-xl" : "bg-slate-100 text-slate-300 cursor-not-allowed"
        }`}>
        {isLoading
          ? <><span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />Running Diagnostic Pipeline…</>
          : <>{valid ? <Zap size={18} className="text-blue-400" /> : <Brain size={18} />}Begin AI Diagnostic Analysis</>}
      </button>
    </form>
  );
}

function Sel({ label, icon, value, onChange, placeholder, options }: {
  label: string; icon: React.ReactNode; value: string; onChange: (v: string) => void;
  placeholder: string; options: { value: string; label: string }[];
}) {
  return (
    <div>
      <p className="label flex items-center gap-1">{icon}{label}</p>
      <div className="relative">
        <select value={value} onChange={e => onChange(e.target.value)} className="input appearance-none pr-8">
          <option value="">{placeholder}</option>
          {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
      </div>
    </div>
  );
}
