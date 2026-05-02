import Header from "@/components/Header";
import Footer from "@/components/Footer";
import BlogCard from "@/components/BlogCard";

// ─── Drop the 6 cover images here before going live ────────────────────────
//   public/images/blog/post-1.jpg  →  Urban Planning
//   public/images/blog/post-2.jpg  →  Transport & Mobility
//   public/images/blog/post-3.jpg  →  Community Impact
//   public/images/blog/post-4.jpg  →  Environment
//   public/images/blog/post-5.jpg  →  Smart City
//   public/images/blog/post-6.jpg  →  Safety & Data
// ───────────────────────────────────────────────────────────────────────────

interface BlogPost {
  id: string;
  category: string;
  title: string;
  description: string;
  image: string;
}

const posts: BlogPost[] = [
  {
    id: "1",
    category: "Urban Planning",
    title:
      "How Melbourne's Pedestrian Counting Network is Shaping Urban Planning",
    description:
      "Melbourne's network of over 70 automated pedestrian counters generates millions of data points each year. Discover how city planners are using this open data...",
    image: "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&q=80",
  },
  {
    id: "2",
    category: "Transport & Mobility",
    title: "Real-Time Public Transport Data: A New Era for Melbourne Commuters",
    description:
      "PTV's open GTFS-RT feed now powers hundreds of third-party apps. We explore how developers are building real-time departure boards, accessibility...",
    image: "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=800&q=80",
  },
  {
    id: "3",
    category: "Community Impact",
    title:
      "Open Data and the Fight Against Food Insecurity in Greater Melbourne",
    description:
      "Community organisations are combining council-published amenity data with census socio-economic indicators to map food deserts across Melbourne's...",
    image: "https://images.unsplash.com/photo-1488521787991-ed7bbaae773c?w=800&q=80",
  },
  {
    id: "4",
    category: "Environment",
    title: "Monitoring Melbourne's Urban Forest: Tree Data Goes Open",
    description:
      "The City of Melbourne's Urban Forest Visual dataset lists every surveyed tree with its species, age, and health rating. Learn how researchers and residents...",
    image: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&q=80",
  },
  {
    id: "5",
    category: "Smart City",
    title: "Sensor-Powered Parking: What Melbourne's Open Parking Data Reveals",
    description:
      "Thousands of on-street sensors feed live bay-occupancy data into Melbourne's open data platform. This post unpacks the patterns hidden in that data...",
    image: "https://images.unsplash.com/photo-1506521781263-d8422e82f27a?w=800&q=80",
  },
  {
    id: "6",
    category: "Safety & Data",
    title:
      "Building a Crime-Aware Cycling Route Planner with Victoria Police Open Data",
    description:
      "By combining Victoria Police's crime statistics, VicRoads crash data, and Melbourne's bicycle network dataset, a team of civic hackers created a...",
    image: "https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=800&q=80",
  },
];

export default function BlogListingPage() {
  return (
    <div className="flex-1 flex flex-col">
      <Header />

      <main className="flex-1 bg-white dark:bg-[#1C1C1C]">
        <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          {/* Page heading */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-3">
              Blog
            </h1>
            <p className="text-sm md:text-base text-gray-500 dark:text-gray-400">
              Insights, updates, and expert tips
            </p>
          </div>

          {/* Responsive card grid: 1 → 2 → 3 columns */}
          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
            {posts.map((post) => (
              <BlogCard
                key={post.id}
                id={post.id}
                title={post.title}
                description={post.description}
                image={post.image}
                category={post.category}
              />
            ))}
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
