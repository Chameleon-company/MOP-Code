'use client'
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from "@/i18n-navigation";
import { useTranslations } from "next-intl";
import { FaFacebook, FaLinkedin } from 'react-icons/fa';
import { FaSquareXTwitter } from "react-icons/fa6";

const Footer = () => {
  const t = useTranslations("common");
  const [hoveredLink, setHoveredLink] = useState<string | null>(null);
  const [parallaxOffset, setParallaxOffset] = useState({ x: 0, y: 0 });
  const footerRef = useRef<HTMLElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const targetParallax = useRef({ x: 0, y: 0 });

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!footerRef.current) return;
    const rect = footerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cx = (x / rect.width - 0.5) * 2;
    const cy = (y / rect.height - 0.5) * 2;
    targetParallax.current = { x: cx * 8, y: cy * 5 };
  }, []);

  useEffect(() => {
    const animate = () => {
      setParallaxOffset(prev => ({
        x: prev.x + (targetParallax.current.x - prev.x) * 0.06,
        y: prev.y + (targetParallax.current.y - prev.y) * 0.06,
      }));
      animFrameRef.current = requestAnimationFrame(animate);
    };
    animFrameRef.current = requestAnimationFrame(animate);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  useEffect(() => {
    const el = footerRef.current;
    if (!el) return;
    el.addEventListener('mousemove', handleMouseMove);
    return () => el.removeEventListener('mousemove', handleMouseMove);
  }, [handleMouseMove]);

  const links = [
    { name: "Licensing", path: "/licensing" },
    { name: "Privacy Policy", path: "/privacypolicy" },
    { name: "Contact Us", path: "/contact" },
  ];

  const socialIcons = [
    { Icon: FaFacebook, label: "Facebook" },
    { Icon: FaSquareXTwitter, label: "Twitter/X" },
    { Icon: FaLinkedin, label: "LinkedIn" },
  ];

  return (
    <>
      <style>{`
        @keyframes gradientShift {
          0%   { background-position: 0% 50%; }
          50%  { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        /* Background shimmer sweep across entire footer */
        @keyframes bgShimmer {
          0%   { transform: translateX(-100%) skewX(-15deg); }
          100% { transform: translateX(200%) skewX(-15deg); }
        }

        @keyframes shimmerSweep {
          0%   { transform: translateX(-150%) skewX(-12deg); }
          100% { transform: translateX(250%) skewX(-12deg); }
        }
        @keyframes floatGlass {
          0%, 100% { transform: translateY(0px) translateX(0px) scale(1); opacity: 0.12; }
          33%       { transform: translateY(-18px) translateX(8px) scale(1.03); opacity: 0.18; }
          66%       { transform: translateY(-8px) translateX(-6px) scale(0.98); opacity: 0.10; }
        }
        @keyframes floatGlass2 {
          0%, 100% { transform: translateY(0px) translateX(0px) scale(1); opacity: 0.08; }
          50%       { transform: translateY(14px) translateX(-10px) scale(1.05); opacity: 0.15; }
        }

        .social-btn {
          position: relative;
          overflow: hidden;
          width: 44px;
          height: 44px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255,255,255,0.08);
          box-shadow:
            4px 4px 10px rgba(0,0,0,0.35),
            -2px -2px 6px rgba(255,255,255,0.12),
            inset 0 1px 0 rgba(255,255,255,0.15);
          border: 1px solid rgba(255,255,255,0.18);
          transition: box-shadow 0.3s ease, background 0.3s ease, color 0.3s ease, transform 0.2s ease;
          cursor: pointer;
          color: rgba(255,255,255,0.85);
        }
        .social-btn:hover {
          background: rgba(255,255,255,0.95);
          color: #166534;
          transform: translateY(-3px) scale(1.08);
          box-shadow:
            0 12px 28px rgba(0,0,0,0.4),
            0 0 20px rgba(255,255,255,0.3),
            inset 0 1px 0 rgba(255,255,255,1);
        }
        .social-btn .shimmer-sweep {
          position: absolute;
          inset: 0;
          background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.7) 50%, transparent 100%);
          transform: translateX(-150%) skewX(-12deg);
          pointer-events: none;
        }
        .social-btn:hover .shimmer-sweep {
          animation: shimmerSweep 0.65s ease forwards;
        }

        .logo-card {
          position: relative;
          overflow: hidden;
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.22);
          border-radius: 20px;
          padding: 18px 26px;
          box-shadow:
            6px 6px 16px rgba(0,0,0,0.4),
            -2px -2px 8px rgba(255,255,255,0.1),
            inset 0 1px 0 rgba(255,255,255,0.2);
          backdrop-filter: blur(12px);
          transition: box-shadow 0.35s ease, transform 0.3s ease;
          cursor: default;
        }
        .logo-card:hover {
          transform: translateY(-4px) scale(1.02);
          box-shadow:
            0 20px 40px rgba(0,0,0,0.45),
            0 0 30px rgba(255,255,255,0.15),
            -2px -2px 8px rgba(255,255,255,0.15),
            inset 0 1px 0 rgba(255,255,255,0.3);
        }
        .logo-card .shimmer-sweep {
          position: absolute;
          inset: 0;
          background: linear-gradient(105deg, transparent 0%, rgba(255,255,255,0.5) 50%, transparent 100%);
          transform: translateX(-150%) skewX(-12deg);
          pointer-events: none;
        }
        .logo-card:hover .shimmer-sweep {
          animation: shimmerSweep 0.9s ease forwards;
        }

        .quick-link {
          position: relative;
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-size: 0.95rem;
          color: rgba(255,255,255,0.92);
          text-decoration: none;
          transition: color 0.2s ease, transform 0.2s ease;
          padding: 2px 0;
          text-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }
        .quick-link::after {
          content: '';
          position: absolute;
          left: 0;
          bottom: -1px;
          width: 0;
          height: 1px;
          background: rgba(255,255,255,0.8);
          transition: width 0.3s ease;
        }
        .quick-link:hover {
          color: #ffffff;
          transform: translateX(5px);
          text-shadow: 0 0 12px rgba(255,255,255,0.6);
        }
        .quick-link:hover::after {
          width: 100%;
        }
        .quick-link .arrow {
          opacity: 0;
          transition: opacity 0.2s ease;
          font-size: 1rem;
        }
        .quick-link:hover .arrow {
          opacity: 1;
        }

        .section-heading {
          font-size: 0.8rem;
          font-weight: 800;
          letter-spacing: 0.22em;
          text-transform: uppercase;
          color: #ffffff;
          text-shadow: 0 0 16px rgba(255,255,255,0.5), 0 1px 4px rgba(0,0,0,0.3);
        }
        .heading-bar {
          height: 3px;
          width: 40px;
          border-radius: 9999px;
          background: rgba(255,255,255,0.9);
          box-shadow: 0 0 8px rgba(255,255,255,0.7), 0 0 16px rgba(255,255,255,0.3);
        }

        /* Mobile fixes */
        @media (max-width: 767px) {
          .footer-col-links {
            border-left: none !important;
            border-right: none !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            align-items: center !important;
          }
          .footer-bottom {
            flex-direction: column !important;
            text-align: center !important;
            gap: 6px !important;
          }
        }
      `}</style>

      <footer
        ref={footerRef}
        className="relative text-white overflow-hidden"
        style={{ minHeight: '380px' }}
        onMouseLeave={() => {
          targetParallax.current = { x: 0, y: 0 };
        }}
      >
        {/* Animated gradient background */}
        <div
          className="absolute inset-0 z-10"
          style={{
            background: 'linear-gradient(135deg, #16a34a 0%, #22c55e 25%, #22c55e 50%, #22c55e 75%, #16a34a 100%)',
            backgroundSize: '400% 400%',
            animation: 'gradientShift 12s ease infinite',
          }}
        />

        {/* Full background shimmer sweep */}
        <div
          className="absolute inset-0 z-10 pointer-events-none"
          style={{
            background: 'linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.12) 50%, transparent 70%)',
            animation: 'bgShimmer 6s ease-in-out infinite',
          }}
        />

        {/* Floating glass layers */}
        <div
          className="absolute rounded-full z-10 pointer-events-none"
          style={{
            width: 340, height: 340, top: '-80px', right: '5%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.13) 0%, transparent 70%)',
            animation: 'floatGlass 9s ease-in-out infinite',
          }}
        />
        <div
          className="absolute rounded-full z-10 pointer-events-none"
          style={{
            width: 220, height: 220, bottom: '-40px', left: '8%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 70%)',
            animation: 'floatGlass2 11s ease-in-out infinite',
          }}
        />
        <div
          className="absolute z-10 pointer-events-none"
          style={{
            width: 180, height: 180, top: '30%', left: '40%',
            borderRadius: '60% 40% 55% 45% / 50% 60% 40% 50%',
            background: 'rgba(255,255,255,0.05)',
            backdropFilter: 'blur(2px)',
            animation: 'floatGlass 14s ease-in-out infinite reverse',
          }}
        />

        {/* Top glow line */}
        <div
          className="absolute top-0 w-full z-20 pointer-events-none"
          style={{
            height: '3px',
            background: 'rgba(255,255,255,0.9)',
            boxShadow: '0 0 12px rgba(255,255,255,0.8), 0 0 24px rgba(255,255,255,0.4)',
          }}
        />

        {/* Content */}
        <div
          className="relative z-20 mx-auto px-6 md:px-8 lg:px-20 pt-14 pb-6"
          style={{
            maxWidth: '1200px',
            transform: `translateX(${parallaxOffset.x * 0.3}px) translateY(${parallaxOffset.y * 0.3}px)`,
            transition: 'transform 0.1s linear',
          }}
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10 items-start">

            {/* BRAND */}
            <div className="flex flex-col items-center md:items-start gap-5">
              <div className="logo-card">
                <img
                  src="/img/new-logo-white.png"
                  alt="Melbourne Open Playground logo"
                  style={{ height: '96px', position: 'relative', zIndex: 1 }}
                />
                <span className="shimmer-sweep" aria-hidden="true" />
              </div>
              <p style={{
                fontSize: '0.9rem',
                color: 'rgba(255,255,255,0.92)',
                lineHeight: '1.7',
                maxWidth: '220px',
                textShadow: '0 1px 4px rgba(0,0,0,0.25)',
              }}
                className="text-center md:text-left">
                Exploring Melbourne&#39;s open data to build smarter communities.
              </p>
            </div>

            {/* QUICK LINKS */}
            <div
              className="footer-col-links flex flex-col items-center md:items-start gap-4"
              style={{
                borderLeft: '1px solid rgba(255,255,255,0.3)',
                borderRight: '1px solid rgba(255,255,255,0.3)',
                paddingLeft: '28px',
                paddingRight: '28px',
              }}
            >
              <p className="section-heading">Quick Links</p>
              <div className="heading-bar" />
              <div className="flex flex-col gap-3 w-full items-center md:items-start">
                {links.map((item, i) => (
                  <Link
                    key={i}
                    href={item.path}
                    className="quick-link"
                    onMouseEnter={() => setHoveredLink(item.name)}
                    onMouseLeave={() => setHoveredLink(null)}
                  >
                    <span className="arrow" aria-hidden="true">›</span>
                    {t(item.name)}
                  </Link>
                ))}
              </div>
            </div>

            {/* CONNECT */}
            <div className="flex flex-col items-center md:items-start gap-4">
              <p className="section-heading">Connect</p>
              <div className="heading-bar" />
              <a
                href="https://data.melbourne.vic.gov.au/pages/home/"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  fontSize: '0.95rem',
                  color: 'rgba(255,255,255,0.92)',
                  textDecoration: 'none',
                  transition: 'color 0.2s ease, text-shadow 0.2s ease',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  textShadow: '0 1px 4px rgba(0,0,0,0.25)',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.color = '#ffffff';
                  e.currentTarget.style.textShadow = '0 0 12px rgba(255,255,255,0.6)';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.color = 'rgba(255,255,255,0.92)';
                  e.currentTarget.style.textShadow = '0 1px 4px rgba(0,0,0,0.25)';
                }}
              >
                Melbourne Open Data
                <span style={{ fontSize: '0.8rem' }}>↗</span>
              </a>
              <div>
                <p style={{
                  fontSize: '0.78rem',
                  color: 'rgba(255,255,255,0.85)',
                  marginBottom: '12px',
                  letterSpacing: '0.12em',
                  textTransform: 'uppercase',
                  textShadow: '0 1px 4px rgba(0,0,0,0.25)',
                  fontWeight: 600,
                }}>
                  Follow us
                  
                </p>
                <div style={{ display: 'flex', gap: '12px' }}>
                  {socialIcons.map(({ Icon, label }, i) => (
                    <button key={i} aria-label={label} className="social-btn">
                      <Icon size={18} style={{ position: 'relative', zIndex: 1 }} />
                      <span className="shimmer-sweep" aria-hidden="true" />
                    </button>
                  ))}
                </div>
              </div>
            </div>

          </div>

          {/* Bottom bar */}
          <div
            className="footer-bottom"
            style={{
              marginTop: '24px',
              paddingTop: '18px',
              borderTop: '1px solid rgba(255,255,255,0.3)',
              textAlign: 'center',
              fontSize: '0.75rem',
              color: 'rgba(255,255,255,0.75)',
              letterSpacing: '0.05em',
              textShadow: '0 1px 3px rgba(0,0,0,0.2)',
            }}
          >
            © {new Date().getFullYear()} Melbourne Open Playground. All rights reserved.
          </div>

        </div>
      </footer>
    </>
  );
};

export default Footer;