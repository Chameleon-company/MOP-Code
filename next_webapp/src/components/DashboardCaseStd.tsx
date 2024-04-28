// DashboardCaseStd.js
import React from 'react';
import Link from "next/link";

const caseStudies = [
    {
        id: 'cs1',
        image: 'https://i.postimg.cc/mg3YLpxm/spreads.png',
        title: 'Case Study 1',
        description: 'Data Science in BioTech',
        link: 'CaseStudy.html'
    },
    {
        id: 'cs2',
        image: 'https://i.postimg.cc/mg3YLpxm/spreads.png',
        title: 'Case Study 2',
        description: 'Predictive Modeling for Maintaining Oil and Gas Supply',
        link: 'casestudy2.html'
    },
    {
        id: 'cs3',
        image: 'https://i.postimg.cc/mg3YLpxm/spreads.png',
        title: 'Case Study 3',
        description: 'Data Science in Education',
        link: 'casestudy3.html'
    }
];

const DashboardCaseStd = () => {
    return (
        <div className='case-studies-wrapper'>
            {caseStudies.map(caseStudy => (
                <Link href={caseStudy.link} key={caseStudy.id}>
                    <div className="card-wrapper">
                        <div className="top-image">
                            <img src={caseStudy.image} alt={caseStudy.title} />
                        </div>
                        <h4 className="title">{caseStudy.title}</h4>
                        <p className="description">{caseStudy.description}</p>
                    </div>
                </Link>
            ))}
        </div>
    );
};

export default DashboardCaseStd;
