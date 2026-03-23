const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Patient {
  patient_id: string;
  name: string;
  dob: string;
  gender: string;
  contact?: string;
}

export interface Physician {
  physician_id: string;
  name: string;
  specialization: string;
  email: string;
}

export interface Technician {
  technician_id: string;
  name: string;
  email: string;
  role: string;
}

export interface DiagnosisOutput {
  diagnosis_label?: string;
  confidence_label?: string;
  severity?: string;
  clinical_note?: string;
  recommended_action?: string;
  differential_diagnoses?: string[];
  class_probabilities?: Record<string, number>;
}

export interface Session {
  session_id: string;
  patient_id: string;
  technician_id: string;
  physician_id: string;
  modality?: string;
  diagnosis_output?: DiagnosisOutput;
  confidence_score?: number;
  clinical_notes?: string;
  model_used?: string;
  explanation?: string;
  report_link?: string;
  status: string;
  created_at?: string;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

export const patientsApi = {
  list: (): Promise<Patient[]> => apiFetch("/patients/"),
  create: (data: Omit<Patient, "patient_id">): Promise<Patient> =>
    apiFetch("/patients/", { method: "POST", body: JSON.stringify(data) }),
};

export const physiciansApi = {
  list: (): Promise<Physician[]> => apiFetch("/physicians/"),
  create: (data: Omit<Physician, "physician_id">): Promise<Physician> =>
    apiFetch("/physicians/", { method: "POST", body: JSON.stringify(data) }),
};

export const techniciansApi = {
  list: (): Promise<Technician[]> => apiFetch("/technicians/"),
  register: (data: { name: string; email: string; password: string; role?: string }): Promise<Technician> =>
    apiFetch("/technicians/register", { method: "POST", body: JSON.stringify(data) }),
};

export const sessionsApi = {
  list: (skip = 0, limit = 20): Promise<Session[]> =>
    apiFetch(`/sessions/?skip=${skip}&limit=${limit}`),

  run: async (
    patientId: string, technicianId: string, physicianId: string,
    modality: string, imageFile: File
  ): Promise<Session> => {
    const form = new FormData();
    form.append("patient_id",    patientId);
    form.append("technician_id", technicianId);
    form.append("physician_id",  physicianId);
    form.append("modality",      modality);
    form.append("image",         imageFile);
    const res = await fetch(`${API_BASE}/sessions/run`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `API error ${res.status}`);
    }
    return res.json();
  },

  heatmapUrl: (id: string) => `${API_BASE}/sessions/${id}/heatmap`,
};
