import BlogSinglePage from "@/components/BLogSInglePage";
import Footer from "@/components/Footer";
import Header from "@/components/Header";

export default async function BlogDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  return (
    <div>
      <Header />
      <BlogSinglePage id={id} />
      <Footer />
    </div>
  );
}
