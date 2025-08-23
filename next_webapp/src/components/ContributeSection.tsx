'use client';

import React, { useCallback, useRef, useState } from 'react';

export default function ContributeSection() {
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const inputRef = useRef<HTMLInputElement>(null);

  const allowedTypes = [
    'text/csv',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/json',
  ];
  const maxSizeMB = 20;

  const validate = (data: FormData) => {
    const e: Record<string, string> = {};
    const required = ['name', 'email', 'title', 'description', 'agree'];
    required.forEach((k) => !data.get(k) && (e[k] = 'This field is required'));
    const email = String(data.get('email') || '');
    if (email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) e.email = 'Enter a valid email';
    if (!file) e.file = 'Please select a dataset file';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleFiles = (files?: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    if (!allowedTypes.includes(f.type)) {
      setMessage('Unsupported file type. Allowed: CSV, XLS, XLSX, JSON.');
      setFile(null);
      return;
    }
    if (f.size > maxSizeMB * 1024 * 1024) {
      setMessage(`File too large. Max ${maxSizeMB} MB.`);
      setFile(null);
      return;
    }
    setMessage(null);
    setFile(f);
  };

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  }, []);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setMessage(null);
    setSubmitting(true);
    const form = e.currentTarget;
    const data = new FormData(form);
    if (!validate(data)) { setSubmitting(false); return; }
    if (file) data.append('file', file);

    try {
      const res = await fetch('/api/contributions', { method: 'POST', body: data });
      if (!res.ok) throw new Error('Upload failed');
      setMessage('✅ Thanks! Your contribution was submitted.');
      form.reset();
      setFile(null);
      setErrors({});
    } catch {
      setMessage('❌ Upload failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="mx-auto max-w-5xl p-6">
      <div className="mb-6 rounded-2xl bg-emerald-900 p-6 text-white">
        <h2 className="text-2xl font-semibold">Contribute Data</h2>
        <p className="mt-1 text-emerald-100">
          Share datasets to grow Melbourne Open Playground. Supported: CSV, XLS/XLSX, JSON.
        </p>
      </div>

      <form onSubmit={onSubmit} className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label htmlFor="name" className="mb-1 block text-sm font-medium">Your Name</label>
          <input id="name" name="name"
            className={`w-full rounded-xl border p-3 focus:outline-none ${errors.name ? 'border-red-500' : 'border-gray-300'}`}
            aria-invalid={!!errors.name}
          />
          {errors.name && <p className="mt-1 text-sm text-red-600">{errors.name}</p>}
        </div>

        <div>
          <label htmlFor="email" className="mb-1 block text-sm font-medium">Email</label>
          <input id="email" name="email" type="email"
            className={`w-full rounded-xl border p-3 focus:outline-none ${errors.email ? 'border-red-500' : 'border-gray-300'}`}
            aria-invalid={!!errors.email}
          />
          {errors.email && <p className="mt-1 text-sm text-red-600">{errors.email}</p>}
        </div>

        <div>
          <label htmlFor="title" className="mb-1 block text-sm font-medium">Dataset Title</label>
          <input id="title" name="title"
            className={`w-full rounded-xl border p-3 focus:outline-none ${errors.title ? 'border-red-500' : 'border-gray-300'}`}
            aria-invalid={!!errors.title}
          />
          {errors.title && <p className="mt-1 text-sm text-red-600">{errors.title}</p>}
        </div>

        <div>
          <label htmlFor="tags" className="mb-1 block text-sm font-medium">Tags (optional)</label>
          <input id="tags" name="tags" className="w-full rounded-xl border border-gray-300 p-3 focus:outline-none" />
        </div>

        <div className="md:col-span-2">
          <label htmlFor="description" className="mb-1 block text-sm font-medium">Short Description</label>
          <textarea id="description" name="description" rows={4}
            className={`w-full rounded-xl border p-3 focus:outline-none ${errors.description ? 'border-red-500' : 'border-gray-300'}`}
            aria-invalid={!!errors.description}
          />
          {errors.description && <p className="mt-1 text-sm text-red-600">{errors.description}</p>}
        </div>

        <div className="md:col-span-2">
          <label className="mb-2 block text-sm font-medium">Dataset File</label>
          <div
            onDragEnter={(e) => { e.preventDefault(); e.stopPropagation(); setDragActive(true); }}
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
            onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); }}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click(); }}
            role="button" tabIndex={0} aria-describedby="file-hint"
            className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center cursor-pointer transition ${dragActive ? 'border-emerald-600 bg-emerald-50' : 'border-gray-300'}`}
          >
            <p className="text-sm" id="file-hint">Drag & drop your file here, or <span className="font-semibold">browse</span></p>
            <p className="mt-1 text-xs text-gray-500">Accepted: CSV, XLS/XLSX, JSON · Max {maxSizeMB} MB</p>
            <input ref={inputRef} type="file" name="file" className="hidden"
              accept=".csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/json"
              onChange={(e) => handleFiles(e.target.files)}
            />
            {file && <div className="mt-3 rounded-full bg-emerald-100 px-3 py-1 text-sm text-emerald-800">Selected: {file.name}</div>}
          </div>
          {errors.file && <p className="mt-1 text-sm text-red-600">{errors.file}</p>}
        </div>

        <div className="md:col-span-2">
          <label className="inline-flex items-start gap-2 text-sm">
            <input type="checkbox" name="agree" className="mt-1" />
            <span>
              I agree to the <a className="underline" href="/contribution-guidelines" target="_blank" rel="noreferrer">contribution guidelines</a> and confirm I have the right to share this data.
            </span>
          </label>
          {errors.agree && <p className="mt-1 text-sm text-red-600">{errors.agree}</p>}
        </div>

        <div className="md:col-span-2 flex items-center gap-3">
          <button type="submit" disabled={submitting}
            className="rounded-2xl bg-emerald-700 px-6 py-3 font-medium text-white hover:bg-emerald-800 disabled:opacity-60">
            {submitting ? 'Submitting…' : 'Submit Contribution'}
          </button>
          {message && <span className="text-sm">{message}</span>}
        </div>
      </form>

      <div className="mt-8 rounded-2xl border bg-emerald-50 p-4">
        <h3 className="text-lg font-semibold">Contribution Tips</h3>
        <ul className="mt-2 list-disc pl-5 text-sm text-emerald-900">
          <li>Use clear titles (e.g., “Playground noise levels – 2024 Q2”).</li>
          <li>Include a short description of columns/units or a README.</li>
          <li>Remove any personal or sensitive information before uploading.</li>
        </ul>
      </div>
    </section>
  );
}
