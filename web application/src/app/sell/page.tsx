"use client";
import { useEffect, useRef, useState } from "react";

const GridCanvas = () => {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    ctx.strokeStyle = "rgba(0,255,200,0.03)";
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 60) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
    for (let y = 0; y < canvas.height; y += 60) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);
  return <canvas ref={ref} className="fixed inset-0 pointer-events-none z-0" />;
};

const Step = ({ n, label, active, done }: { n: number; label: string; active: boolean; done: boolean }) => (
  <div className="flex items-center gap-2">
    <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold shrink-0"
      style={{
        background: done ? "#00ffc8" : active ? "rgba(0,255,200,0.15)" : "rgba(255,255,255,0.04)",
        border: `1px solid ${done ? "#00ffc8" : active ? "rgba(0,255,200,0.5)" : "rgba(255,255,255,0.1)"}`,
        color: done ? "#000" : active ? "#00ffc8" : "rgba(255,255,255,0.3)",
      }}>
      {done ? "✓" : n}
    </div>
    <span className="text-xs font-mono tracking-widest uppercase"
      style={{ color: active ? "rgba(255,255,255,0.8)" : done ? "#00ffc8" : "rgba(255,255,255,0.25)" }}>
      {label}
    </span>
  </div>
);

const Divider = () => (
  <div className="w-8 h-px mx-1" style={{ background: "rgba(255,255,255,0.1)" }} />
);

const UploadBox = ({ file, onFile }: { file: File | null; onFile: (f: File) => void }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const preview = file ? URL.createObjectURL(file) : null;

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) onFile(f); }}
      className="relative w-full h-52 rounded-xl cursor-pointer overflow-hidden flex items-center justify-center transition-all duration-300"
      style={{
        border: `1px dashed ${drag ? "#00ffc8" : "rgba(0,255,200,0.2)"}`,
        background: drag ? "rgba(0,255,200,0.04)" : "rgba(0,10,20,0.4)",
        boxShadow: drag ? "0 0 20px rgba(0,255,200,0.1)" : "none",
      }}>
      <input ref={inputRef} type="file" accept="image/*" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={preview} alt="preview" className="w-full h-full object-contain" />
      ) : (
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl mx-auto mb-3 flex items-center justify-center"
            style={{ background: "rgba(0,255,200,0.08)", border: "1px solid rgba(0,255,200,0.2)" }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00ffc8" strokeWidth="1.5">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
          </div>
          <p className="font-mono text-xs text-white/40">Drop your NFT image here</p>
          <p className="font-mono text-xs text-white/20 mt-1">PNG, JPG, WEBP</p>
        </div>
      )}
    </div>
  );
};

const ScannerOverlay = () => (
  <div className="absolute inset-0 overflow-hidden rounded-xl pointer-events-none">
    <div className="absolute left-0 right-0 h-0.5"
      style={{
        background: "linear-gradient(90deg, transparent, #00ffc8, transparent)",
        animation: "scanY 2s ease-in-out infinite",
        boxShadow: "0 0 12px #00ffc8",
      }} />
  </div>
);

type VerifyResult = { match: boolean; extracted: string; wallet: string; bits_accuracy: number };

const ResultCard = ({ result, onReset }: { result: VerifyResult; onReset: () => void }) => {
  const getConfidence = (acc: number) => {
    if (acc >= 90) return {
      title: "Ownership Verified",
      subtitle: "Watermark strongly matches the provided wallet address",
      color: "#00ffc8", bg: "rgba(0,255,200,0.06)", border: "rgba(0,255,200,0.3)",
    };
    if (acc >= 70) return {
      title: "Likely Your NFT",
      subtitle: "Partial watermark match - this is probably the original owner's NFT",
      color: "#60a5fa", bg: "rgba(96,165,250,0.06)", border: "rgba(96,165,250,0.3)",
    };
    if (acc >= 50) return {
      title: "Possible Match",
      subtitle: "Weak signal — image may have been compressed or edited",
      color: "#fbbf24", bg: "rgba(251,191,36,0.06)", border: "rgba(251,191,36,0.3)",
    };
    return {
      title: "No Match Detected",
      subtitle: "Watermark does not match this wallet — or image was heavily modified",
      color: "#f87171", bg: "rgba(248,113,113,0.06)", border: "rgba(248,113,113,0.3)",
    };
  };

  const confidence = getConfidence(result.bits_accuracy);

  return (
    <div className="space-y-4">
      <div className="rounded-xl p-6 text-center"
        style={{ background: confidence.bg, border: `1px solid ${confidence.border}` }}>
        <div className="font-mono text-lg font-bold mb-1" style={{ color: confidence.color }}>
          {confidence.title}
        </div>
        <div className="font-mono text-xs text-white/40">{confidence.subtitle}</div>
      </div>

      <div className="rounded-xl p-4" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
        <div className="flex justify-between items-center mb-2">
          <span className="font-mono text-xs text-white/30 uppercase tracking-widest">Watermark Confidence</span>
          <span className="font-mono text-xs font-bold" style={{ color: confidence.color }}>
            {result.bits_accuracy.toFixed(1)}%
          </span>
        </div>
        <div className="w-full h-1.5 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
          <div className="h-full rounded-full transition-all duration-700"
            style={{ width: `${result.bits_accuracy}%`, background: confidence.color }} />
        </div>
        <div className="relative mt-1 h-3">
          <div className="absolute font-mono text-white/15" style={{ left: "50%", fontSize: "9px" }}>50%</div>
          <div className="absolute font-mono text-white/15" style={{ left: "70%", fontSize: "9px" }}>70%</div>
          <div className="absolute font-mono text-white/15" style={{ left: "90%", fontSize: "9px" }}>90%</div>
        </div>
      </div>

      {result.bits_accuracy < 90 && (
        <div className="rounded-xl p-4 font-mono text-xs text-white/25 space-y-1"
          style={{ background: "rgba(255,255,255,0.01)", border: "1px solid rgba(255,255,255,0.04)" }}>
          <div className="text-white/40 mb-1">Why isn't this definitive?</div>
          <div>· Image compression or resizing can degrade the watermark signal</div>
          <div>· Model accuracy improves with more training data</div>
          <div>· Cross-reference with on-chain ownership for full certainty</div>
        </div>
      )}

      <button onClick={onReset}
        className="w-full py-3 rounded-xl font-mono text-sm transition-all"
        style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.6)" }}>
        Verify Another
      </button>
    </div>
  );
};

export default function VerifyPage() {
  const [wallet, setWallet] = useState<string>("");
  const [walletError, setWalletError] = useState<string | null>(null);

  const [mounted, setMounted] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [step, setStep] = useState(1);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { setMounted(true); }, []);

  useEffect(() => {
    if (!wallet || wallet.length < 10) { setStep(1); return; }
    if (!file) { setStep(2); return; }
    setStep(3);
  }, [wallet, file]);

  if (!mounted) return null;

  const reset = () => {
    setFile(null); setResult(null); setError(null); setScanning(false);
  };

  const handleVerify = async () => {
    if (!file || !wallet) return;
    setScanning(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("wallet_address", wallet);

      const res = await fetch("https://polar-expire-postage.ngrok-free.dev/verify", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Verification request failed");
      const data = await res.json();
      setResult(data);
      setStep(3);
    } catch (err) {
      setError("Could not connect to the verification backend. Make sure your FastAPI server is running.");
    } finally {
      setScanning(false);
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@400;600&display=swap');
        @keyframes fadeUp {
          from { opacity:0; transform:translateY(20px); }
          to   { opacity:1; transform:translateY(0); }
        }
        @keyframes scanY {
          0%   { top: 0%; }
          50%  { top: 95%; }
          100% { top: 0%; }
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .fade-up { animation: fadeUp 0.6s ease forwards; }
        .btn-cyan {
          background: linear-gradient(135deg,#00ffc8,#00b8a0);
          color:#000; font-weight:600;
          transition: box-shadow .3s, transform .2s;
        }
        .btn-cyan:hover { box-shadow:0 0 28px rgba(0,255,200,.45); transform:translateY(-2px); }
        .btn-cyan:disabled { opacity:.4; pointer-events:none; }
        .card {
          background: rgba(0,15,30,.65);
          border: 1px solid rgba(0,255,200,.1);
          backdrop-filter: blur(14px);
          border-radius: 20px;
        }
        .spinner {
          width:18px; height:18px; border:2px solid rgba(0,0,0,.2);
          border-top-color:#000; border-radius:50%;
          animation: spin .8s linear infinite; display:inline-block;
        }
      `}</style>

      <div className="relative min-h-screen" style={{ background: "#060d1a", fontFamily: "'DM Sans',sans-serif" }}>
        <GridCanvas />

        <div className="fixed top-1/3 right-1/4 w-[500px] h-[300px] pointer-events-none rounded-full z-0"
          style={{ background: "radial-gradient(ellipse,rgba(0,100,255,0.05) 0%,transparent 70%)" }} />
        <div className="fixed bottom-1/4 left-1/4 w-[400px] h-[300px] pointer-events-none rounded-full z-0"
          style={{ background: "radial-gradient(ellipse,rgba(0,255,200,0.04) 0%,transparent 70%)" }} />

        <div className="relative z-10 pt-28 pb-20 px-6 max-w-2xl mx-auto">

          <div className="fade-up mb-10">
            <div className="font-mono text-xs text-cyan-500/50 tracking-widest uppercase mb-2">Verify Ownership</div>
            <h1 className="text-3xl font-bold mb-2"
              style={{ fontFamily: "'DM Mono',monospace", background: "linear-gradient(135deg,#fff 0%,#a0f0e0 60%,#00ffc8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Extract & Verify
            </h1>
            <p className="text-white/40 text-sm leading-relaxed">
              Paste a wallet address, upload a watermarked NFT, and we'll extract the hidden signature to confirm whether that address is the rightful owner.
            </p>
          </div>

          <div className="fade-up flex items-center mb-8" style={{ animationDelay: "0.1s", opacity: 0 }}>
            <Step n={1} label="Wallet Address" active={step === 1} done={step > 1} />
            <Divider />
            <Step n={2} label="Upload NFT" active={step === 2} done={step > 2} />
            <Divider />
            <Step n={3} label="Verify" active={step === 3} done={result?.match === true} />
          </div>

          <div className="card p-7 fade-up" style={{ animationDelay: "0.15s", opacity: 0 }}>

            <div className="mb-6">
              <label className="font-mono text-xs text-white/40 uppercase tracking-widest block mb-3">
                01 · Wallet Address
              </label>
              <input
                type="text"
                placeholder="0x..."
                value={wallet}
                onChange={(e) => {
                  setWallet(e.target.value);
                  setWalletError(null);
                }}
                className="w-full px-4 py-3 rounded-xl font-mono text-sm text-cyan-300 outline-none transition-all"
                style={{
                  background: "rgba(0,255,200,0.06)",
                  border: `1px solid ${walletError ? "rgba(248,113,113,0.5)" : "rgba(0,255,200,0.2)"}`,
                }}
              />
              {walletError && (
                <p className="font-mono text-xs mt-1" style={{ color: "#f87171" }}>{walletError}</p>
              )}
            </div>

            <div className="mb-6" style={{ opacity: wallet ? 1 : 0.35, pointerEvents: wallet ? "auto" : "none", transition: "opacity .3s" }}>
              <label className="font-mono text-xs text-white/40 uppercase tracking-widest block mb-3">
                02 · Watermarked NFT Image
              </label>
              <div className="relative">
                <UploadBox file={file} onFile={(f) => { setFile(f); setResult(null); setError(null); }} />
                {scanning && file && <ScannerOverlay />}
              </div>
              {file && !scanning && (
                <p className="font-mono text-xs text-white/30 mt-2">
                  {file.name} · {(file.size / 1024).toFixed(1)} KB
                </p>
              )}
            </div>

            <div style={{ opacity: wallet && file ? 1 : 0.35, pointerEvents: wallet && file ? "auto" : "none", transition: "opacity .3s" }}>
              <label className="font-mono text-xs text-white/40 uppercase tracking-widest block mb-3">
                03 · Result
              </label>

              {result ? (
                <ResultCard result={result} onReset={reset} />
              ) : (
                <div className="space-y-3">
                  {!scanning && !error && (
                    <div className="rounded-xl p-4 font-mono text-xs text-white/30 space-y-1.5"
                      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <div>① Watermark extracted from image via decoder</div>
                      <div>② Extracted bits converted back to address</div>
                      <div>③ Compared against the provided wallet address</div>
                    </div>
                  )}

                  {error && (
                    <div className="rounded-xl p-4 font-mono text-xs"
                      style={{ background: "rgba(248,113,113,0.06)", border: "1px solid rgba(248,113,113,0.2)", color: "#f87171" }}>
                      {error}
                    </div>
                  )}

                  <button
                    onClick={handleVerify}
                    disabled={scanning || !wallet || !file}
                    className="btn-cyan w-full py-4 rounded-xl font-mono text-sm tracking-wide flex items-center justify-center gap-3">
                    {scanning ? (
                      <><span className="spinner" /> Scanning for watermark…</>
                    ) : (
                      "Extract & Verify Ownership →"
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>

          <p className="text-center font-mono text-xs text-white/20 mt-6 fade-up" style={{ animationDelay: "0.3s", opacity: 0 }}>
            Extraction runs on your image only · No data is stored
          </p>
        </div>
      </div>
    </>
  );
}
