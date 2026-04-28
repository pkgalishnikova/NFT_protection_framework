// "use client";
// import Link from "next/link";

// export function Navbar() {
//   return (
//     <>
//       <style>{`
//         @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;600&display=swap');
//         .nav-link {
//           font-family: 'DM Mono', monospace;
//           font-size: 0.75rem;
//           letter-spacing: 0.05em;
//           color: rgba(255,255,255,0.4);
//           transition: color 0.2s ease;
//           position: relative;
//           padding-bottom: 2px;
//         }
//         .nav-link::after {
//           content: '';
//           position: absolute;
//           bottom: -2px;
//           left: 0;
//           width: 0;
//           height: 1px;
//           background: #00ffc8;
//           transition: width 0.25s ease;
//         }
//         .nav-link:hover {
//           color: #00ffc8;
//         }
//         .nav-link:hover::after {
//           width: 100%;
//         }
//         .nav-logo-mark {
//           width: 32px; height: 32px; border-radius: 8px;
//           background: rgba(0,255,200,0.08);
//           border: 1px solid rgba(0,255,200,0.25);
//           display: flex; align-items: center; justify-content: center;
//           transition: box-shadow 0.3s ease;
//         }
//         .nav-logo-mark:hover { box-shadow: 0 0 16px rgba(0,255,200,0.3); }
//         .nav-cta {
//           font-family: 'DM Mono', monospace;
//           font-size: 0.75rem; letter-spacing: 0.1em; font-weight: 600;
//           padding: 8px 18px; border-radius: 10px;
//           background: linear-gradient(135deg, #00ffc8, #00b8a0);
//           color: #000;
//           transition: box-shadow 0.3s ease, transform 0.2s ease;
//         }
//         .nav-cta:hover { box-shadow: 0 0 20px rgba(0,255,200,0.4); transform: translateY(-1px); }
//           font-family: 'DM Mono', monospace;
//           font-size: 0.6rem;
//           letter-spacing: 0.15em;
//           color: #00ffc8;
//           border: 1px solid rgba(0,255,200,0.3);
//           padding: 1px 6px;
//           border-radius: 4px;
//           background: rgba(0,255,200,0.05);
//         }
//       `}</style>

//       <div
//         className="fixed top-0 z-50 w-full"
//         style={{
//           background: "rgba(6, 13, 26, 0.75)",
//           backdropFilter: "blur(20px)",
//           borderBottom: "1px solid rgba(0,255,200,0.08)",
//         }}
//       >
//         <nav className="flex items-center justify-between w-full px-8 py-4 mx-auto max-w-7xl">
//           <Link href="/" className="flex items-center gap-3 group">
//             <div className="nav-logo-mark">
//               <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
//                 <rect x="2" y="2" width="14" height="14" rx="2" stroke="#00ffc8" strokeWidth="1.2" strokeDasharray="3 1.5" />
//                 <circle cx="9" cy="9" r="3" fill="none" stroke="#00ffc8" strokeWidth="1.2" />
//                 <circle cx="9" cy="9" r="1" fill="#00ffc8" />
//               </svg>
//             </div>
//             <div className="flex flex-col leading-tight">
//               <span className="nav-logo-text">StegaVerify</span>
//             </div>
//             <span className="nav-badge hidden sm:inline">BETA</span>
//           </Link>

//           <div className="flex items-center gap-8">
//             <Link href="/buy" className="nav-link uppercase tracking-widest">
//               Embed
//             </Link>
//             <Link href="/sell" className="nav-link uppercase tracking-widest">
//               Verify
//             </Link>
//             <Link href="/buy" className="nav-cta">
//               Get Started
//             </Link>
//           </div>
//         </nav>
//       </div>
//     </>
//   );
// }