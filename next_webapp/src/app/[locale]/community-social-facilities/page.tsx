import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import React from "react";
import FacilitySection from "../../../components/facilities/FacilitySection";
import { communitySectionsYouOwn } from "../../../data/facilities";

export const metadata = {
  title: "Community & Social Facilities",
};

export default function Page() {
  return (
    <>
      <Header />
      <main>
      {communitySectionsYouOwn.map((sec) => (
        <FacilitySection key={sec.id} {...sec} />
      ))}
    </main>
      <Footer />
    </>
  );
}
