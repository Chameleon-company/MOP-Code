import React from "react";
import styles from "./facilities.module.css";
import FacilityCard from "./FacilityCard";
import type { FacilitySectionData } from "../../types/facilities";

export default function FacilitySection({
  id,
  title,
  subtitle,
  items,
  tone = "green",
}: FacilitySectionData) {
  return (
    <section id={id} className={styles.section}>
      <div className={`${styles.panel} ${tone === "blue" ? styles.panelBlue : styles.panelGreen}`}>
        {/* Title + subtitle INSIDE the colored panel, tight spacing */}
        <div className={styles.panelHeader}>
          <h2 className={styles.panelTitle}>{title}</h2>
          <p className={styles.panelSubtitle}>{subtitle}</p>
        </div>

        {/* Cards grid */}
        <div className={styles.grid}>
          {items.map((it) => (
            <FacilityCard key={it.name} {...it} />
          ))}
        </div>
      </div>
    </section>
  );
}

