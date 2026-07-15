import Image from "next/image";
import { Link } from "@/i18n-navigation";

interface BlogCardProps {
  id: string;
  title: string;
  description: string;
  image: string;
  category: string;
}

export default function BlogCard({
  id,
  title,
  description,
  image,
  category,
}: BlogCardProps) {
  return (
    <Link href={`/blog/${id}`} className="group block h-full">
      <div className="h-full bg-white dark:bg-gray-800 rounded-xl shadow-md hover:shadow-lg transition-shadow duration-300 overflow-hidden flex flex-col">
        {/* Cover image — 16:9 aspect ratio */}
        <div className="relative aspect-video overflow-hidden flex-shrink-0">
          <Image
            src={image}
            alt={title}
            fill
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
            className="object-cover group-hover:scale-105 transition-transform duration-300"
          />
          {/* Category badge — bottom-left overlay */}
          <span className="absolute bottom-2 left-2 bg-black/60 text-white text-xs font-medium px-2.5 py-1 rounded-full backdrop-blur-sm">
            {category}
          </span>
        </div>

        {/* Card body */}
        <div className="p-5 flex flex-col flex-grow">
          <h3 className="text-base font-bold text-gray-900 dark:text-white line-clamp-2 mb-2 group-hover:text-green-500 transition-colors duration-200 leading-snug">
            {title}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 line-clamp-3 flex-grow mb-5 leading-relaxed">
            {description}
          </p>
          {/* Read More — outlined green, hover fills */}
          <span className="self-start border border-green-500 text-green-500 text-sm font-medium px-4 py-1.5 rounded-lg group-hover:bg-green-500 group-hover:text-white transition-colors duration-200">
            Read More →
          </span>
        </div>
      </div>
    </Link>
  );
}
