

"use client";

import { useEffect, useState } from "react";
import Image from "next/image";

type Testimonial = {
  quote: string;
  organization: string;
  logo?: string;
};

const testimonials: Testimonial[] = [
  {
    quote:
      "This platform helped us present our work more clearly and connect with the right audience.",
    organization: "Deakin University",
    logo: "/partners/user.png",
  },
  {
    quote:
      "A clean and practical solution that made collaboration and content sharing much easier. The workflow felt more organized, and the structure made it easier for different teams to stay aligned during the project lifecycle.",
    organization: "Lenovo",
    logo: "/partners/user.png",
  },
  {
    quote:
      "The overall experience was smooth, modern, and genuinely useful for our team. The interface was easy to understand, and the responsiveness across devices made a big difference for day-to-day use.",
    organization: "User",
    logo: "/partners/user.png",
  },
  {
    quote:
      "A thoughtful project with strong usability and a professional feel throughout. It showed attention to detail, both in design and in the way information was presented to users.",
    organization: "Energy Australia",
    logo: "/partners/user.png",
  },
  {
    quote:
      "Highly intuitive and well-designed. It made our workflow much more efficient and reduced the time needed to explain the platform to new users joining the project.",
    organization: "UFC",
    logo: "/partners/user.png",
  },
];

export default function TestimonialsSection() {
  const [index, setIndex] = useState(0);
  const [selectedReview, setSelectedReview] = useState<Testimonial | null>(null);

  const total = testimonials.length;

  const nextSlide = () => {
    setIndex((prev) => (prev + 1) % total);
  };

  const prevSlide = () => {
    setIndex((prev) => (prev - 1 + total) % total);
  };

  useEffect(() => {
    const interval = setInterval(() => {
      nextSlide();
    }, 4000);

    return () => clearInterval(interval);
  }, [index]);

  const getVisibleTestimonials = () => {
    return [
      testimonials[index],
      testimonials[(index + 1) % total],
      testimonials[(index + 2) % total],
    ];
  };

  const visible = getVisibleTestimonials();

  return (
    <>
      <section className="w-full bg-gray-50 dark:bg-[#1f2a30] py-20 px-4">
        <div className="max-w-6xl mx-auto text-center mb-14">
          <h2 className="text-4xl font-bold text-gray-800 dark:text-white mb-3">
            Testimonials
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            What our collaborators say about this project.
          </p>
        </div>

        <div className="max-w-6xl mx-auto relative">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 pt-8">
            {visible.map((item, i) => (
              <button
                key={`${item.organization}-${i}`}
                type="button"
                onClick={() => setSelectedReview(item)}
                className="group relative rounded-3xl border border-gray-200/70 dark:border-white/10 bg-white/95 dark:bg-[#263238]/95 px-6 pt-14 pb-6 text-center shadow-[0_10px_30px_rgba(0,0,0,0.06)] transition-all duration-500 hover:-translate-y-2 hover:shadow-[0_18px_45px_rgba(0,0,0,0.12)] min-h-[320px] flex flex-col items-center cursor-pointer backdrop-blur-sm"
              >
                <div className="absolute inset-0 rounded-3xl opacity-0 group-hover:opacity-100 transition duration-500 bg-gradient-to-br from-emerald-100/40 via-transparent to-transparent dark:from-emerald-500/10 pointer-events-none" />

                {item.logo && (
                  <div className="absolute -top-7 left-1/2 -translate-x-1/2 z-10">
                    <div className="h-16 w-16 rounded-full bg-white dark:bg-[#1f2a30] shadow-lg ring-4 ring-white dark:ring-[#263238]">
                      <Image
                        src={item.logo}
                        alt={item.organization}
                        width={60}
                        height={60}
                        className="h-full w-full rounded-full object-cover"
                      />
                    </div>
                  </div>
                )}

                <p
                  className="text-gray-700 dark:text-gray-200 italic text-[15px] leading-8 mb-6 overflow-hidden"
                  style={{
                    display: "-webkit-box",
                    WebkitLineClamp: 5,
                    WebkitBoxOrient: "vertical",
                  }}
                >
                  “{item.quote}”
                </p>

                <h4 className="font-semibold text-gray-900 dark:text-white text-base mt-auto">
                  {item.organization}
                </h4>

                <span className="mt-3 text-sm text-emerald-600 dark:text-emerald-400 font-medium opacity-80 group-hover:opacity-100 transition">
                  Read more
                </span>
              </button>
            ))}
          </div>

          <div className="flex items-center justify-center gap-4 mt-10">
            <button
              onClick={prevSlide}
              className="h-11 w-11 rounded-full bg-white dark:bg-[#263238] border border-gray-200 dark:border-white/10 shadow-md text-gray-700 dark:text-white hover:scale-105 transition"
              aria-label="Previous testimonials"
            >
              ←
            </button>

            <div className="flex gap-2">
              {testimonials.map((_, dotIndex) => (
                <button
                  key={dotIndex}
                  onClick={() => setIndex(dotIndex)}
                  className={`h-3 w-3 rounded-full transition-all duration-300 ${
                    dotIndex === index
                      ? "bg-emerald-500 scale-110"
                      : "bg-gray-300 dark:bg-gray-500"
                  }`}
                  aria-label={`Go to testimonial set ${dotIndex + 1}`}
                />
              ))}
            </div>

            <button
              onClick={nextSlide}
              className="h-11 w-11 rounded-full bg-white dark:bg-[#263238] border border-gray-200 dark:border-white/10 shadow-md text-gray-700 dark:text-white hover:scale-105 transition"
              aria-label="Next testimonials"
            >
              →
            </button>
          </div>
        </div>
      </section>

      {selectedReview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-[3px] px-4"
          onClick={() => setSelectedReview(null)}
        >
          <div
            className="relative w-full max-w-2xl rounded-3xl border border-gray-200/80 dark:border-white/10 bg-white dark:bg-[#263238] p-8 md:p-10 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setSelectedReview(null)}
              className="absolute top-4 right-4 h-10 w-10 rounded-full bg-gray-100 dark:bg-[#1f2a30] text-gray-600 dark:text-gray-300 hover:scale-105 transition"
              aria-label="Close review popup"
            >
              ×
            </button>

            {selectedReview.logo && (
              <div className="flex justify-center mb-6">
                <div className="h-20 w-20 rounded-full bg-white dark:bg-[#1f2a30] shadow-lg ring-4 ring-gray-100 dark:ring-[#1f2a30]">
                  <Image
                    src={selectedReview.logo}
                    alt={selectedReview.organization}
                    width={72}
                    height={72}
                    className="h-full w-full rounded-full object-cover"
                  />
                </div>
              </div>
            )}

            <p className="text-gray-700 dark:text-gray-200 italic text-lg leading-8 mb-6 text-center">
              “{selectedReview.quote}”
            </p>

            <h4 className="text-center font-semibold text-lg text-gray-900 dark:text-white">
              {selectedReview.organization}
            </h4>
          </div>
        </div>
      )}
    </>
  );
}