"use client";

import Image from "next/image";

const partners = [
  { name: "UFC", logo: "/partners/ufc.png" },
  { name: "X", logo: "/partners/x.png" },
  { name: "Deakin", logo: "/partners/deakin.png" },
  { name: "Lenovo", logo: "/partners/lenovo.png" },
  { name: "Energy", logo: "/partners/energy.png" },
  { name: "Amazon", logo: "/partners/amazon.png" },
  { name: "Shell", logo: "/partners/shell.png" },
  { name: "Melbourne", logo: "/partners/melbourne.png" },
  { name: "Post", logo: "/partners/post.png" },
];

export default function PartnersSection() {
  const repeatedPartners = [...partners, ...partners];

  return (
    <section className="w-full bg-white dark:bg-[#263238] py-20 overflow-hidden bottom-padding " >
      <div className="max-w-6xl mx-auto px-4 text-center mb-10">
        <h2 className="text-4xl font-bold text-gray-800 dark:text-white mb-3">
          Our Partners
        </h2>
        <p className="text-gray-600 dark:text-gray-300">
          Organizations supporting and collaborating with this project.
        </p>
      </div>

      <div className="relative max-w-7xl mx-auto overflow-hidden px-4">
        <div className="pointer-events-none absolute inset-y-0 left-0 w-20 bg-gradient-to-r from-white dark:from-[#263238] to-transparent z-10" />
        <div className="pointer-events-none absolute inset-y-0 right-0 w-20 bg-gradient-to-l from-white dark:from-[#263238] to-transparent z-10" />

        <div className="marquee">
          <div className="marquee-track">
            {repeatedPartners.map((partner, index) => (
              <div key={index} className="logo-card group">
                <Image
                  src={partner.logo}
                  alt={partner.name}
                  width={250}
                  height={125}
                  className="max-h-[80px] w-auto object-contain opacity-70 transition-all duration-300 group-hover:grayscale-0 group-hover:opacity-100"
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      <style jsx>{`
        .marquee {
          overflow: hidden;
          width: 100%;
        }

        .marquee-track {
          display: flex;
          align-items: center;
          width: max-content;
          animation: scroll 42s linear infinite;
        }

        .marquee-track:hover {
          animation-play-state: paused;
        }
        .bottom-padding{
        
        padding-bottom:120px;
        
        }
        .logo-card {
          flex: 0 0 auto;
          width: 190px;
          height: 110px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-right: 20px;
          border-radius: 18px;
          background: rgba(249, 250, 251, 1);
          padding: 24px;
          box-shadow: none !important;
          border: 1px solid #e5e7eb;
        }

        :global(.dark) .logo-card {
          background: #2f3b42;
          box-shadow: none !important;
          border: 1px solid #3f4d55;
        }

        @keyframes scroll {
          from {
            transform: translateX(0);
          }
          to {
            transform: translateX(-50%);
          }
        }
      `}</style>
    </section>
  );
}
