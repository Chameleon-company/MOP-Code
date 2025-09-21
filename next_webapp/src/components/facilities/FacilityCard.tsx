import React from "react";
import styles from "./facilities.module.css";

type Props = {
  name: string;
  href: string;
  image: string;
  alt: string;
};

export default function FacilityCard({ name, href, image, alt }: Props) {
  return (
    <a className={styles.card} href={href} target="_blank" rel="noopener noreferrer">
      <div className={styles.imageWrap}>
        <img className={styles.image} src={image} alt={alt} loading="lazy" />
      </div>
      <div className={styles.cardTitle}>{name}</div>
    </a>
  );
}



