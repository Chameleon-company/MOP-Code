import FoodServicesCaseBody from "../../../../components/FoodServicesCaseBody";

export default function FoodServicesPage() {
  return (
    <main className="bg-[#0e1621] min-h-screen">
      <FoodServicesCaseBody
        heroSrc="/P1.png" // put P1,P2,P3 in next_webapp/public/
        charts={{ barSrc: "/P2.png", lineSrc: "/P3.png" }}
        wordCloudSrc="/P3.png"
      />
    </main>
  );
}

