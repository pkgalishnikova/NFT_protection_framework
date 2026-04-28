"use client";
import type { NextPage } from "next";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

const ParticleCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles: {
      x: number; y: number; vx: number; vy: number;
      size: number; opacity: number; pulse: number;
    }[] = [];

    for (let i = 0; i < 80; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 2 + 0.5,
        opacity: Math.random() * 0.6 + 0.1,
        pulse: Math.random() * Math.PI * 2,
      });
    }

    let animId: number;
    let frame = 0;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      frame++;

      ctx.strokeStyle = "rgba(0, 255, 200, 0.04)";
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 60) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 60) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(0, 220, 180, ${0.12 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      particles.forEach((p) => {
        p.pulse += 0.02;
        const glow = Math.sin(p.pulse) * 0.3 + 0.7;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 255, 200, ${p.opacity * glow})`;
        ctx.fill();
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
      });

      animId = requestAnimationFrame(draw);
    };

    draw();

    const onResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(animId); window.removeEventListener("resize", onResize); };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
};

const FloatingImage = () => {
  const [tick, setTick] = useState(0);
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 50);
    return () => clearInterval(id);
  }, []);

  useEffect(() => { setMounted(true); }, []);

  const rows = 8;
  const cols = 12;
  return (
    <div className="relative w-full max-w-lg mx-auto select-none">
      <div className="absolute inset-0 rounded-2xl blur-3xl opacity-30"
        style={{ background: "radial-gradient(ellipse, #00ffc8 0%, transparent 70%)" }} />

      <div className="relative rounded-2xl border border-cyan-500/30 overflow-hidden shadow-2xl"
        style={{ background: "linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%)" }}>

        <div className="absolute inset-0 grid opacity-60"
          style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gridTemplateRows: `repeat(${rows}, 1fr)` }}>
          {mounted && Array.from({ length: rows * cols }).map((_, i) => {  // WRAP WITH mounted &&
            const active = Math.sin(i * 0.7 + tick * 0.08) > 0.6;
            return (
              <div key={i} className="transition-all duration-300"
                style={{
                  background: active
                    ? `rgba(0, 255, 200, ${(Math.sin(i * 1.3) * 0.15 + 0.2)})`  // REPLACE Math.random()
                    : "transparent",
                  border: "0.5px solid rgba(0,255,200,0.05)"
                }} />
            );
          })}
        </div>

        <div className="relative z-10 flex flex-col items-center justify-center py-16 gap-4">
          <div className="relative">
            <div className="w-24 h-24 rounded-xl border-2 border-cyan-400/60 flex items-center justify-center"
              style={{ background: "rgba(0,255,200,0.08)", boxShadow: "0 0 40px rgba(0,255,200,0.2)" }}>
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <rect x="6" y="6" width="36" height="36" rx="4" stroke="#00ffc8" strokeWidth="1.5" strokeDasharray="4 2" />
                <circle cx="24" cy="24" r="8" fill="none" stroke="#00ffc8" strokeWidth="1.5" />
                <circle cx="24" cy="24" r="3" fill="#00ffc8" />
                <line x1="24" y1="6" x2="24" y2="16" stroke="#00ffc8" strokeWidth="1.5" />
                <line x1="24" y1="32" x2="24" y2="42" stroke="#00ffc8" strokeWidth="1.5" />
                <line x1="6" y1="24" x2="16" y2="24" stroke="#00ffc8" strokeWidth="1.5" />
                <line x1="32" y1="24" x2="42" y2="24" stroke="#00ffc8" strokeWidth="1.5" />
              </svg>
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-28 h-28 rounded-full border border-cyan-500/20 animate-spin"
                style={{ animationDuration: "4s" }}>
                <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 absolute -top-1 left-1/2 -translate-x-1/2"
                  style={{ boxShadow: "0 0 8px #00ffc8" }} />
              </div>
            </div>
          </div>

          <div className="w-48 h-px relative overflow-hidden rounded-full"
            style={{ background: "rgba(0,255,200,0.1)" }}>
            <div className="absolute h-full w-12 rounded-full"
              style={{
                background: "linear-gradient(90deg, transparent, #00ffc8, transparent)",
                animation: "scan 2s ease-in-out infinite",
              }} />
          </div>

          <div className="font-mono text-xs text-cyan-400/60 tracking-widest">
            WATERMARK DETECTED
          </div>
        </div>

        {["top-2 left-2", "top-2 right-2", "bottom-2 left-2", "bottom-2 right-2"].map((pos, i) => (
          <div key={i} className={`absolute ${pos} w-4 h-4`}>
            <div className="w-full h-0.5 bg-cyan-400/60" />
            <div className={`w-0.5 h-full bg-cyan-400/60 ${i % 2 === 0 ? "ml-0" : "ml-auto"}`} />
          </div>
        ))}
      </div>

      <div className="mt-4 font-mono text-xs text-cyan-500/30 text-center leading-relaxed overflow-hidden h-8">
        <div style={{ animation: "slideUp 3s linear infinite" }}>
          {["0x4a8f2c...e91b", "ADDRESS · MATCH", "0x7d3e1a...f28c", "VERIFIED ✓"].map((t, i) => (
            <div key={i}>{t}</div>
          ))}
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, delay }: { label: string; value: string; delay: string }) => (
  <div className="border border-cyan-900/50 rounded-xl p-5 text-center"
    style={{
      background: "rgba(0,20,40,0.6)",
      backdropFilter: "blur(10px)",
      animation: `fadeUp 0.8s ease forwards`,
      animationDelay: delay,
      opacity: 0,
    }}>
    <div className="font-mono text-2xl font-bold text-cyan-300 mb-1">{value}</div>
    <div className="text-xs text-white/40 tracking-widest uppercase">{label}</div>
  </div>
);

const StepCard = ({ num, title, desc, delay }: { num: string; title: string; desc: string; delay: string }) => (
  <div className="flex gap-4 items-start"
    style={{ animation: `fadeUp 0.6s ease forwards`, animationDelay: delay, opacity: 0 }}>
    <div className="font-mono text-3xl font-bold text-cyan-500/30 leading-none w-10 shrink-0">{num}</div>
    <div>
      <div className="text-white font-semibold mb-1" style={{ fontFamily: "'DM Sans', sans-serif" }}>{title}</div>
      <div className="text-white/40 text-sm leading-relaxed">{desc}</div>
    </div>
  </div>
);

const Home: NextPage = () => {
  const [visible, setVisible] = useState(false);
  useEffect(() => { setTimeout(() => setVisible(true), 100); }, []);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;600;700&display=swap');
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(24px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes scan {
          0% { left: -50%; }
          100% { left: 150%; }
        }
        @keyframes slideUp {
          0% { transform: translateY(0); }
          100% { transform: translateY(-100%); }
        }
        @keyframes glowPulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 0.8; }
        }
        .hero-title {
          font-family: 'DM Mono', monospace;
          background: linear-gradient(135deg, #ffffff 0%, #a0f0e0 40%, #00ffc8 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .btn-primary {
          background: linear-gradient(135deg, #00ffc8, #00b8a0);
          color: #000;
          font-weight: 600;
          transition: all 0.3s ease;
        }
        .btn-primary:hover {
          box-shadow: 0 0 30px rgba(0,255,200,0.5);
          transform: translateY(-2px);
        }
        .btn-secondary {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.1);
          color: rgba(255,255,255,0.7);
          transition: all 0.3s ease;
        }
        .btn-secondary:hover {
          background: rgba(255,255,255,0.08);
          transform: translateY(-2px);
        }
      `}</style>

      <div className="relative min-h-screen overflow-x-hidden"
        style={{ background: "#060d1a", fontFamily: "'DM Sans', sans-serif" }}>

        <ParticleCanvas />

        <div className="fixed top-0 left-1/4 w-96 h-96 rounded-full pointer-events-none z-0"
          style={{ background: "radial-gradient(circle, rgba(0,255,200,0.06) 0%, transparent 70%)", animation: "glowPulse 4s ease infinite" }} />
        <div className="fixed bottom-1/4 right-1/4 w-80 h-80 rounded-full pointer-events-none z-0"
          style={{ background: "radial-gradient(circle, rgba(0,100,255,0.05) 0%, transparent 70%)", animation: "glowPulse 6s ease infinite 2s" }} />

        <main className="relative z-10 pt-32 pb-20 px-6">
          <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">

            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-cyan-500/30 mb-8 font-mono text-xs text-cyan-400/80"
                style={{
                  background: "rgba(0,255,200,0.05)",
                  animation: "fadeUp 0.6s ease forwards",
                  opacity: 0
                }}>
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                Blockchain · Neural Watermarking · ETH Verification
              </div>

              <h1 className="hero-title text-5xl lg:text-6xl font-bold leading-tight mb-6"
                style={{ animation: "fadeUp 0.7s ease 0.1s forwards", opacity: 0 }}>
                Prove NFT<br />Ownership.<br />
                <span style={{ fontWeight: 300 }}>Invisibly.</span>
              </h1>

              <p className="text-white/50 text-lg leading-relaxed mb-10 max-w-md"
                style={{ animation: "fadeUp 0.7s ease 0.2s forwards", opacity: 0, fontFamily: "'DM Sans', sans-serif" }}>
                Embed an address of your wallet directly into your NFT image using neural steganography. Verify ownership without ever exposing your key.
              </p>

              <div className="flex gap-4 flex-wrap relative" style={{ zIndex: 20 }}>
                <Link href="/buy" className="btn-primary px-7 py-3.5 rounded-xl text-sm font-mono tracking-wide">
                  Embed Watermark
                </Link>
                <Link href="/sell" className="btn-secondary px-7 py-3.5 rounded-xl text-sm font-mono tracking-wide">
                  Verify Ownership
                </Link>
              </div>
            </div>

            <div style={{ animation: "fadeUp 0.9s ease 0.3s forwards", opacity: 0 }}>
              <FloatingImage />
            </div>
          </div>

          <div className="max-w-7xl mx-auto mt-32">
            <div className="text-center mb-16"
              style={{ animation: "fadeUp 0.7s ease 0.2s forwards", opacity: 0 }}>
              <div className="font-mono text-xs text-cyan-500/60 tracking-widest mb-3 uppercase">Protocol</div>
              <h2 className="text-3xl font-bold text-white">How it works</h2>
            </div>

            <div className="grid lg:grid-cols-2 gap-16 items-start">
              <div className="rounded-2xl p-8 border border-cyan-900/40"
                style={{ background: "rgba(0,15,30,0.6)", backdropFilter: "blur(12px)" }}>
                <div className="font-mono text-xs text-cyan-400/60 tracking-widest mb-6 uppercase">01 / Embed</div>
                <div className="flex flex-col gap-6">
                  <StepCard num="1" title="Enter your wallet address" desc="Connect your wallet, so the system can learn your wallet address for embedding." delay="0.1s" />
                  <div className="w-px h-4 bg-cyan-900/50 ml-4" />
                  <StepCard num="2" title="Upload your NFT image" desc="Provide the original image you plan to mint your NFT with." delay="0.2s" />
                  <div className="w-px h-4 bg-cyan-900/50 ml-4" />
                  <StepCard num="3" title="Download watermarked image" desc="Model embeds the hash invisibly into the image. Visually almost identical — cryptographically bound." delay="0.3s" />
                </div>
              </div>

              <div className="rounded-2xl p-8 border border-cyan-900/40"
                style={{ background: "rgba(0,15,30,0.6)", backdropFilter: "blur(12px)" }}>
                <div className="font-mono text-xs text-cyan-400/60 tracking-widest mb-6 uppercase">02 / Verify</div>
                <div className="flex flex-col gap-6">
                  <StepCard num="1" title="Provide your wallet address" desc="Connect your wallet, so the system recieves your wallet address for comparison." delay="0.1s" />
                  <div className="w-px h-4 bg-cyan-900/50 ml-4" />
                  <StepCard num="2" title="Upload the NFT image" desc="Submit the watermarked image to be checked for an embedded ownership signature." delay="0.2s" />
                  <div className="w-px h-4 bg-cyan-900/50 ml-4" />
                  <StepCard num="3" title="Instant verification result" desc="The extracted watermark is compared to your address. Match = verified owner. No match = invalid claim." delay="0.3s" />
                </div>
              </div>
            </div>
          </div>

          <div className="max-w-3xl mx-auto mt-24 text-center rounded-2xl p-12 border border-cyan-500/20"
            style={{
              background: "linear-gradient(135deg, rgba(0,255,200,0.05), rgba(0,100,255,0.05))",
              backdropFilter: "blur(12px)",
              animation: "fadeUp 0.7s ease 0.2s forwards",
              opacity: 0
            }}>
            <div className="font-mono text-xs text-cyan-400/60 tracking-widest mb-4 uppercase">Ready to start?</div>
            <h3 className="text-3xl font-bold text-white mb-4">Your NFT. Your proof.</h3>
            <p className="text-white/40 mb-8 text-sm max-w-md mx-auto">
              Ownership verification that lives inside the image itself — no metadata, no external registry.
            </p>
            <Link href="/buy" className="btn-primary inline-block px-10 py-4 rounded-xl font-mono text-sm tracking-wide">
              Get Started →
            </Link>
          </div>
        </main>

        <footer className="relative z-10 border-t border-white/5 py-8 px-6 text-center">
          <p className="font-mono text-xs text-white/20">
            NFT Watermark Verification Framework · Diploma Project · Ethereum
          </p>
        </footer>
      </div>
    </>
  );
};

export default Home;