"use client";

import {
    BarElement,
    CategoryScale,
    Chart as ChartJS,
    Colors,
    Legend,
    LinearScale,
    Title,
    Tooltip,
} from "chart.js";
import { Bar } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface StatisticsChartProps {
    data: any;
    options: any;
}

export default function StatisticsChart({ data, options }: StatisticsChartProps) {
    return <Bar data={data} options={options} />;
}