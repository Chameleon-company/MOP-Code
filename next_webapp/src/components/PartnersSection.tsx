// "use client";

// import Image from "next/image";

// const partners = [
//   { name: "Deakin University", logo: "/partners/deakin.png" },
//   { name: "City of Melbourne", logo: "/partners/melbourne.png" },
//   { name: "Data.gov.au", logo: "/partners/data-gov.png" },
//   { name: "Open Data Victoria", logo: "/partners/open-data.png" },
// ];

// export default function PartnersSection() {
//   return (
//     <section className="w-full bg-white dark:bg-[#263238] py-14">
//       <div className="max-w-6xl mx-auto px-4 text-center">
//         <h2 className="text-3xl font-bold text-gray-800 dark:text-white mb-3">
//           Our Partners
//         </h2>
//         <p className="text-gray-600 dark:text-gray-300 mb-10">
//           Organizations supporting and collaborating with this project.
//         </p>

//         <div className="grid grid-cols-2 md:grid-cols-4 gap-6 items-center">
//           {partners.map((partner, index) => (
//             <div
//               key={index}
//               className="bg-gray-50 dark:bg-gray-800 rounded-2xl shadow-sm p-6 flex items-center justify-center hover:shadow-md transition duration-300"
//             >
//               <Image
//                 src={partner.logo}
//                 alt={partner.name}
//                 width={140}
//                 height={70}
//                 className="object-contain grayscale hover:grayscale-0 transition duration-300"
//               />
//             </div>
//           ))}
//         </div>
//       </div>
//     </section>
//   );
// }


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
    <section className="w-full bg-white dark:bg-[#263238] py-14 overflow-hidden">
      <div className="max-w-6xl mx-auto px-4 text-center mb-8">
        <h2 className="text-4xl font-bold text-gray-800 dark:text-white mb-3">
          Our Partners
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          Organizations supporting and collaborating with this project.
        </p>
      </div>

      <div className="marquee mt-4">
        <div className="marquee-track">
          {repeatedPartners.map((partner, index) => (
            <div key={index} className="logo-item">
              <Image
                src={partner.logo}
                alt={partner.name}
                width={160}
                height={80}
                className="logo-img"
              />
            </div>
          ))}

          {repeatedPartners.map((partner, index) => (
            <div key={`dup-${index}`} className="logo-item">
              <Image
                src={partner.logo}
                alt={partner.name}
                width={160}
                height={80}
                className="logo-img"
              />
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .marquee {
          width: 100%;
          overflow: hidden;
          position: relative;
        }

        .marquee-track {
          display: flex;
          align-items: center;
          width: max-content;
          animation: scroll 25s linear infinite;
        }

        .logo-item {
          flex: 0 0 auto;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0 32px;
        }

        .logo-img {
          width: auto;
          height: 70px;
          object-fit: contain;
        }

        .marquee-track:hover {
          animation-play-state: paused;
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