"use client";
import React from "react";
import { Link } from "@/i18n-navigation";
import { useCases } from "@/utils/data";

const Insights: React.FC = () => {
	return (
		<section
			className="w-full bg-white dark:bg-[#263238] py-12 px-6 text-black dark:text-white"
			aria-labelledby="usecases-heading"
		>
			<div className="text-center mb-10">
				<h2
					id="usecases-heading"
					className="text-3xl md:text-4xl font-bold mb-2"
				>
					Explore Our Use Cases
				</h2>
				<p className="text-gray-600 dark:text-gray-300 text-sm md:text-base">
					Discover how our innovative solutions transform industries worldwide.
				</p>
			</div>

			<div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
				{useCases.map((useCase) => (
					<Link
						key={useCase.id}
						href={`/usecases/${useCase.id}`}
						className="bg-gray-50 dark:bg-[#37474F] rounded-2xl shadow-md hover:shadow-lg transition p-4 flex flex-col group"
					>
						<img
							src={useCase.image}
							alt={useCase.title}
							className="rounded-xl mb-4 w-full h-40 object-cover group-hover:scale-[1.02] transition-transform"
						/>
						<h3 className="text-xl font-semibold mb-2">{useCase.title}</h3>
						<p className="text-gray-600 dark:text-gray-300 text-sm flex-grow">
							{useCase.description}
						</p>
						<span className="mt-4 bg-green-500 group-hover:bg-green-600 text-white py-2 px-4 rounded-xl text-sm font-medium text-center">
							View Details →
						</span>
					</Link>
				))}
			</div>
		</section>
	);
};

export default Insights;
