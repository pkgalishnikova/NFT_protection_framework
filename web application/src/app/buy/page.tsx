"use client";
import { useEffect, useRef, useState } from "react";
import { client } from "../client";
import { prepareContractCall, sendTransaction } from "thirdweb";
import { useActiveAccount, useSendTransaction, ConnectButton } from "thirdweb/react";
import { verifyNFTContract } from "../../contract";

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
        <img src={preview} alt="preview" className="w-full h-full object-contain" />
      ) : (
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl mx-auto mb-3 flex items-center justify-center"
            style={{ background: "rgba(0,255,200,0.08)", border: "1px solid rgba(0,255,200,0.2)" }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00ffc8" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p className="font-mono text-xs text-white/40">Drop image here or click to browse</p>
          <p className="font-mono text-xs text-white/20 mt-1">PNG, JPG, WEBP</p>
        </div>
      )}
    </div>
  );
};

const StatusBadge = ({ status }: { status: "idle" | "embedding" | "minting" | "done" | "error" }) => {
  const map = {
    idle: { label: "Ready", color: "rgba(255,255,255,0.3)", bg: "rgba(255,255,255,0.04)" },
    embedding: { label: "Embedding watermark…", color: "#00ffc8", bg: "rgba(0,255,200,0.08)" },
    minting: { label: "Minting NFT…", color: "#60a5fa", bg: "rgba(96,165,250,0.08)" },
    done: { label: "Success", color: "#00ffc8", bg: "rgba(0,255,200,0.08)" },
    error: { label: "Error — try again", color: "#f87171", bg: "rgba(248,113,113,0.08)" },
  };
  const s = map[status];
  return (
    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full font-mono text-xs"
      style={{ background: s.bg, border: `1px solid ${s.color}33`, color: s.color }}>
      {(status === "embedding" || status === "minting") && (
        <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: s.color }} />
      )}
      {s.label}
    </div>
  );
};

export default function EmbedMintPage() {
  const account = useActiveAccount();
  const wallet = account?.address ?? null;

  const [mounted, setMounted] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "embedding" | "minting" | "done" | "error">("idle");
  const [txHash, setTxHash] = useState<string | null>(null);
  const [watermarkedUrl, setWatermarkedUrl] = useState<string | null>(null);
  const [step, setStep] = useState(1);
  const reset = () => {
    setFile(null);
    setStatus("idle");
    setTxHash(null);
    setWatermarkedUrl(null);
  };

  useEffect(() => { setMounted(true); }, []);

  useEffect(() => {
    if (!wallet) { setStep(1); return; }
    if (!file) { setStep(2); return; }
    setStep(3);
  }, [wallet, file]);

  if (!mounted) return null;



  const handleEmbedAndMint = async () => {
    if (!file || !wallet || !account) return;
    const activeAccount = account;

    try {
      setStatus("embedding");
      const formData = new FormData();
      formData.append("image", file);
      formData.append("wallet_address", wallet);

      const embedRes = await fetch("https://polar-expire-postage.ngrok-free.dev/embed", {
        method: "POST", body: formData,
      });
      if (!embedRes.ok) throw new Error("Embedding failed");
      const embedBlob = await embedRes.blob();
      setWatermarkedUrl(URL.createObjectURL(embedBlob));

      setStatus("minting");
      const ipfsForm = new FormData();
      ipfsForm.append("image", embedBlob, "watermarked.png");
      ipfsForm.append("name", file.name.replace(/\.[^.]+$/, "") || "My NFT");

      const ipfsRes = await fetch("https://polar-expire-postage.ngrok-free.dev/upload-ipfs", {
        method: "POST", body: ipfsForm,
      });
      if (!ipfsRes.ok) throw new Error("IPFS upload failed");
      const { token_uri } = await ipfsRes.json();

      const transaction = prepareContractCall({
        contract: verifyNFTContract,
        method: "function mint(string tokenURI_, string embeddedAddr) returns (uint256)",
        params: [token_uri, wallet],
      });

      const receipt = await sendTransaction({
        transaction,
        account: activeAccount,
      });

      setTxHash(receipt.transactionHash);
      setStatus("done");


    } catch (err) {
      console.error(err);
      setStatus("error");
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
        @keyframes spin { to { transform: rotate(360deg); } }
        .fade-up { animation: fadeUp 0.6s ease forwards; }
        .btn-cyan {
          background: linear-gradient(135deg,#00ffc8,#00b8a0);
          color:#000; font-weight:600;
          transition: box-shadow .3s, transform .2s;
        }
        .btn-cyan:hover { box-shadow:0 0 28px rgba(0,255,200,.45); transform:translateY(-2px); }
        .btn-cyan:disabled { opacity:.4; pointer-events:none; }
        .btn-ghost {
          background: rgba(255,255,255,.04);
          border: 1px solid rgba(255,255,255,.1);
          color: rgba(255,255,255,.6);
          transition: background .2s;
        }
        .btn-ghost:hover { background: rgba(255,255,255,.08); }
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

        <div className="fixed top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[300px] pointer-events-none rounded-full"
          style={{ background: "radial-gradient(ellipse,rgba(0,255,200,0.04) 0%,transparent 70%)" }} />

        <div className="relative z-10 pt-28 pb-20 px-6 max-w-2xl mx-auto">

          <div className="fade-up mb-10">
            <div className="font-mono text-xs text-cyan-500/50 tracking-widest uppercase mb-2">Embed & Mint</div>
            <h1 className="text-3xl font-bold mb-2"
              style={{ fontFamily: "'DM Mono',monospace", background: "linear-gradient(135deg,#fff 0%,#a0f0e0 60%,#00ffc8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Watermark your NFT
            </h1>
            <p className="text-white/40 text-sm leading-relaxed">
              Connect your wallet, upload an image, and we'll embed your address as an invisible watermark before minting.
            </p>
          </div>

          <div className="fade-up flex items-center mb-8" style={{ animationDelay: "0.1s", opacity: 0 }}>
            <Step n={1} label="Connect Wallet" active={step === 1} done={step > 1} />
            <Divider />
            <Step n={2} label="Upload Image" active={step === 2} done={step > 2} />
            <Divider />
            <Step n={3} label="Embed & Mint" active={step === 3} done={status === "done"} />
          </div>

          <div className="card p-7 fade-up" style={{ animationDelay: "0.15s", opacity: 0 }}>

            <div className="mb-6">
              <label className="font-mono text-xs text-white/40 uppercase tracking-widest block mb-3">
                01 · Wallet Address
              </label>
              {wallet ? (
                <div className="flex items-center justify-between px-4 py-3 rounded-xl"
                  style={{ background: "rgba(0,255,200,0.06)", border: "1px solid rgba(0,255,200,0.2)" }}>
                  <div>
                    <div className="font-mono text-xs text-cyan-400/60 mb-0.5">Connected</div>
                    <div className="font-mono text-sm text-cyan-300">
                      {wallet.slice(0, 6)}…{wallet.slice(-4)}
                    </div>
                  </div>
                  <ConnectButton client={client} />
                </div>
              ) : (
                <ConnectButton
                  client={client}
                  connectButton={{
                    label: "Connect Wallet",
                    className: "btn-cyan w-full py-3.5 rounded-xl font-mono text-sm tracking-wide !w-full",
                  }}
                />
              )}
            </div>

            <div className="mb-6" style={{ opacity: wallet ? 1 : 0.35, pointerEvents: wallet ? "auto" : "none", transition: "opacity .3s" }}>
              <label className="font-mono text-xs text-white/40 uppercase tracking-widest block mb-3">
                02 · NFT Image
              </label>
              <UploadBox file={file} onFile={setFile} />
              {file && (
                <p className="font-mono text-xs text-white/30 mt-2">
                  {file.name} · {(file.size / 1024).toFixed(1)} KB
                </p>
              )}
            </div>

            <div style={{ opacity: wallet && file ? 1 : 0.35, pointerEvents: wallet && file ? "auto" : "none", transition: "opacity .3s" }}>
              <label className="font-mono text-xs text-white/40 uppercase tracking-widest block mb-3">
                03 · Embed & Mint
              </label>

              {status === "done" ? (
                <div className="space-y-4">
                  <div className="rounded-xl p-5 text-center"
                    style={{ background: "rgba(0,255,200,0.06)", border: "1px solid rgba(0,255,200,0.25)" }}>
                    <div className="text-2xl mb-2">✅</div>
                    <div className="font-mono text-sm text-cyan-300 font-bold mb-1">NFT Minted Successfully</div>
                    <div className="font-mono text-xs text-white/40">Watermark embedded · Address bound on-chain</div>
                    {txHash && (
                      <div className="mt-3 font-mono text-xs text-white/30 break-all">
                        TX: {txHash.slice(0, 18)}…{txHash.slice(-6)}
                      </div>
                    )}
                  </div>

                  {watermarkedUrl && (
                    <a href={watermarkedUrl} download="watermarked-nft.png"
                      className="btn-ghost w-full py-3 rounded-xl font-mono text-sm text-center block">
                      ↓ Download Watermarked Image
                    </a>
                  )}

                  <button onClick={reset}
                    className="btn-ghost w-full py-3 rounded-xl font-mono text-sm">
                    Mint Another
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <StatusBadge status={status} />
                    {status === "error" && (
                      <button onClick={() => setStatus("idle")}
                        className="font-mono text-xs text-white/30 hover:text-white/60">
                        Reset
                      </button>
                    )}
                  </div>

                  {status === "idle" && (
                    <div className="rounded-xl p-4 font-mono text-xs text-white/30 space-y-1.5"
                      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <div>① Your wallet address will be collected</div>
                      <div>② Address embedded invisibly into your image</div>
                      <div>③ Watermarked image uploaded to IPFS</div>
                      <div>④ NFT minted to your wallet via smart contract</div>
                    </div>
                  )}

                  <button
                    onClick={handleEmbedAndMint}
                    disabled={status === "embedding" || status === "minting"}
                    className="btn-cyan w-full py-4 rounded-xl font-mono text-sm tracking-wide flex items-center justify-center gap-3">
                    {(status === "embedding" || status === "minting") ? (
                      <><span className="spinner" /> {status === "embedding" ? "Embedding watermark…" : "Minting on-chain…"}</>
                    ) : (
                      "Embed Watermark & Mint NFT →"
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
} 
