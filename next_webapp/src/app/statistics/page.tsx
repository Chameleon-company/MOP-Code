"use client"
import React, { useState, useEffect } from 'react';
import Header from "../../components/Header";
import Footer from "../../components/Footer";
//import caseStudies from './database';  // Adjust the path according to your project structure


const Statistics = () => {
    // Dummy Array
    const caseStudies = [
        { id: 1, tag: 'Safety and Well-being', publishNumber: '4', popularity: '11%'},
        { id: 2, tag: 'Environment and Sustainability', publishNumber: '5', popularity: '20%'},
        { id: 3, tag: 'Business and activity', publishNumber: '8', popularity: '90%'}
        
    ];
    

    // State for storing the filtered results and all filters
    const [filteredStudies, setFilteredStudies] = useState(caseStudies);
    const [tagFilter, setTagFilter] = useState('');
    const [publishFilter, setPublishFilter] = useState('');
    const [popularityFilter, setPopularityFilter] = useState('');

    // Distinct tags for the dropdown
    const tags = Array.from(new Set(caseStudies.map(study => study.tag)));

    // Effect to handle filtering based on tag, publish number, and popularity
    useEffect(() => {
        let filtered = caseStudies;
        if (tagFilter) {
            filtered = filtered.filter(study => study.tag === tagFilter);
        }
        if (publishFilter) {
            filtered = filtered.filter(study => {
                const publishNum = parseInt(study.publishNumber, 10);
                switch (publishFilter) {
                    case 'lessThan4':
                        return publishNum < 4;
                    case 'between4And7':
                        return publishNum >= 4 && publishNum <= 7;
                    case 'above7':
                        return publishNum > 7;
                    default:
                        return true;
                }
            });
        }
        if (popularityFilter) {
            filtered = filtered.filter(study => {
                const popularity = parseFloat(study.popularity.replace('%', ''));
                switch (popularityFilter) {
                    case '0to20':
                        return popularity > 0 && popularity <= 20;
                    case '20to40':
                        return popularity > 20 && popularity <= 40;
                    case '40to60':
                        return popularity > 40 && popularity <= 60;
                    case '60to80':
                        return popularity > 60 && popularity <= 80;
                    case '80to100':
                        return popularity > 80 && popularity <= 100;
                    default:
                        return true;
                }
            });
        }
        setFilteredStudies(filtered);
        
    }, [tagFilter, publishFilter, popularityFilter]);

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }} className="font-sans bg-gray-100 text-black">
            <Header />
            <main style={{ flex: '1 0 auto', width: '100%' }}>
                <div>
                    
                    <section>
                        <select 
                            value={tagFilter} 
                            onChange={(e) => setTagFilter(e.target.value)} 
                            className="p-2 m-2 border"
                        >
                            <option value="">All Tags</option>
                            {tags.map(tag => (
                                <option key={tag} value={tag}>
                                    {tag}
                                </option>
                            ))}
                        </select>
                        <select
                            value={publishFilter}
                            onChange={(e) => setPublishFilter(e.target.value)}
                            className="p-2 m-2 border"
                        >
                            <option value="">All Publish Numbers</option>
                            <option value="lessThan4">Less than 4</option>
                            <option value="between4And7">Between 4 and 7</option>
                            <option value="above7">Above 7</option>
                        </select>
                        <select
                            value={popularityFilter}
                            onChange={(e) => setPopularityFilter(e.target.value)}
                            className="p-2 m-2 border"
                        >
                            <option value="">All Popularity Ranges</option>
                            <option value="0to20">0 - 20%</option>
                            <option value="20to40">20 - 40%</option>
                            <option value="40to60">40 - 60%</option>
                            <option value="60to80">60 - 80%</option>
                            <option value="80to100">80 - 100%</option>
                        </select>
                        <div className="overflow-hidden rounded-lg shadow">
                            <table className="min-w-full bg-white">
                                <thead>
                                    <tr>
                                        <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tl-lg">
                                            ID
                                        </th>
                                        <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tl-lg">
                                            Tag
                                        </th>
                                        <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                                            Number of Case Studies Published
                                        </th>
                                        <th className="px-5 py-3 border-b-2 border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider rounded-tr-lg">
                                            Popularity
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredStudies.map((study, index) => (
                                        <tr key={study.id} className={index % 2 === 0 ? 'bg-gray-100' : 'bg-white'}>
                                            <td className="px-5 py-5 border-b border-gray-200 text-sm">
                                                {study.id}
                                            </td>
                                            <td className="px-5 py-5 border-b border-gray-200 text-sm">
                                                {study.tag}
                                            </td>
                                            <td className="px-5 py-5 border-b border-gray-200 text-sm">
                                                {study.publishNumber}
                                            </td>
                                            <td className="px-5 py-5 border-b border-gray-200 text-sm">
                                                {study.popularity}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </section>
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default Statistics;
