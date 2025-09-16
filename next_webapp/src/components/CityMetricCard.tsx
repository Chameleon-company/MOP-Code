"use client";
import React from "react";
import {
  TrendingUp,
  TrendingDown,
  Users,
  Car,
  Home,
  DollarSign,
  Heart,
  Trees,
} from "lucide-react";

export interface CityMetric {
  id: string;
  title: string;
  value: string | number;
  change: number;
  icon: React.ReactNode;
  category:
    | "population"
    | "transportation"
    | "environment"
    | "housing"
    | "economy"
    | "health";
}

interface CityMetricCardProps {
  metric: CityMetric;
  className?: string;
}

const CityMetricCard: React.FC<CityMetricCardProps> = ({
  metric,
  className = "",
}) => {
  const isPositive = metric?.change >= 0;

  // Define colors based on category
  const getCategoryColors = () => {
    switch (metric.category) {
      case "population":
        return {
          bg: "bg-blue-100",
          text: "text-blue-700",
          darkBg: "dark:bg-blue-900/30",
          darkText: "dark:text-blue-300",
        };
      case "transportation":
        return {
          bg: "bg-purple-100",
          text: "text-purple-700",
          darkBg: "dark:bg-purple-900/30",
          darkText: "dark:text-purple-300",
        };
      case "environment":
        return {
          bg: "bg-green-100",
          text: "text-green-700",
          darkBg: "dark:bg-green-900/30",
          darkText: "dark:text-green-300",
        };
      case "housing":
        return {
          bg: "bg-amber-100",
          text: "text-amber-700",
          darkBg: "dark:bg-amber-900/30",
          darkText: "dark:text-amber-300",
        };
      case "economy":
        return {
          bg: "bg-indigo-100",
          text: "text-indigo-700",
          darkBg: "dark:bg-indigo-900/30",
          darkText: "dark:text-indigo-300",
        };
      case "health":
        return {
          bg: "bg-pink-100",
          text: "text-pink-700",
          darkBg: "dark:bg-pink-900/30",
          darkText: "dark:text-pink-300",
        };
      default:
        return {
          bg: "bg-gray-100",
          text: "text-gray-700",
          darkBg: "dark:bg-gray-900/30",
          darkText: "dark:text-gray-300",
        };
    }
  };

  const colors = getCategoryColors();

  return (
    <div
      className={`rounded-xl p-5 shadow-md transition-all duration-300 hover:shadow-lg bg-white dark:bg-gray-800 overflow-hidden relative ${className}`}
    >
      {/* Background pattern */}
      <div
        className={`absolute -right-4 -top-4 w-24 h-24 rounded-full opacity-10 ${colors.bg} ${colors.darkBg}`}
      ></div>

      <div className="flex justify-between items-start mb-4 relative z-10">
        <div>
          <h3 className="font-semibold text-gray-600 dark:text-gray-300 text-sm uppercase tracking-wide">
            {metric.title}
          </h3>
          <p className="text-2xl font-bold text-gray-800 dark:text-white mt-1">
            {metric.value}
          </p>
        </div>
        <div className={`p-3 rounded-lg ${colors.bg} ${colors.darkBg}`}>
          {metric.icon}
        </div>
      </div>

      <div className="flex items-center mt-4">
        <div
          className={`flex items-center ${
            isPositive
              ? "text-green-600 dark:text-green-400"
              : "text-red-600 dark:text-red-400"
          }`}
        >
          {isPositive ? (
            <TrendingUp size={16} className="mr-1" />
          ) : (
            <TrendingDown size={16} className="mr-1" />
          )}
          <span className="text-sm font-medium">
            {isPositive ? "+" : ""}
            {metric.change}%
          </span>
        </div>
        <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
          from last year
        </span>
      </div>

      {/* Progress bar */}
      <div className="mt-3">
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
          <div
            className={`h-1.5 rounded-full ${
              isPositive
                ? "bg-green-500 dark:bg-green-400"
                : "bg-red-500 dark:bg-red-400"
            }`}
            style={{ width: `${Math.min(Math.abs(metric.change), 100)}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

// Sample data and component usage example
export const CityMetricsDashboard: React.FC = () => {
  const sampleMetrics: CityMetric[] = [
    {
      id: "1",
      title: "Population",
      value: "2.3M",
      change: 2.5,
      icon: <Users size={20} className="text-blue-700 dark:text-blue-300" />,
      category: "population",
    },
    {
      id: "2",
      title: "Public Transport",
      value: "78%",
      change: 5.2,
      icon: <Car size={20} className="text-purple-700 dark:text-purple-300" />,
      category: "transportation",
    },
    {
      id: "3",
      title: "Green Spaces",
      value: "32%",
      change: -1.2,
      icon: <Trees size={20} className="text-green-700 dark:text-green-300" />,
      category: "environment",
    },
    {
      id: "4",
      title: "Housing Affordability",
      value: "64%",
      change: 3.1,
      icon: <Home size={20} className="text-amber-700 dark:text-amber-300" />,
      category: "housing",
    },
    {
      id: "5",
      title: "Median Income",
      value: "$65,420",
      change: 4.7,
      icon: (
        <DollarSign
          size={20}
          className="text-indigo-700 dark:text-indigo-300"
        />
      ),
      category: "economy",
    },
    {
      id: "6",
      title: "Life Expectancy",
      value: "81.2 yrs",
      change: 0.8,
      icon: <Heart size={20} className="text-pink-700 dark:text-pink-300" />,
      category: "health",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 p-6 bg-gray-50 dark:bg-gray-900 min-h-screen">
      {sampleMetrics.map((metric) => (
        <CityMetricCard key={metric.id} metric={metric} />
      ))}
    </div>
  );
};

export default CityMetricCard;
