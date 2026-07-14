import { CaseStudy, CATEGORY } from "../../types";

export const demoCaseStudies: CaseStudy[] = [
  {
    id: 1,
    title: "Smart Parking System",
    description:
      "An AI-powered parking solution that helps users find available parking spots in real time using IoT sensors.",
    tags: ["AI", "IoT", "Smart City"],
    htmlFile: "/usecases/use-case-1.html",
    category:  "Smart City",
    image: "/usecases/parking.jpg",
  },
  {
    id: 2,
    title: "Intelligent Waste Management",
    description:
      "A smart waste collection system that optimizes garbage pickup routes using sensor data and analytics.",
    tags: ["Sustainability", "IoT", "Analytics"],
    htmlFile: "/usecases/use-case-2.html",
    category: "Waste Management",
    image: "/usecases/waste.jpg",
  },
  {
    id: 3,
    title: "Real-Time Traffic Monitoring",
    description:
      "A system that analyzes live traffic conditions and suggests optimal routes using machine learning.",
    tags: ["Machine Learning", "Transport", "Real-Time"],
    htmlFile: "/usecases/use-case-3.html",
    category: "Traffic Devlopment",
    image: "/usecases/traffic.jpg",
  },
];