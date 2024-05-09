// DashboardCaseStd.js
import React from 'react';
import Link from "next/link";


const caseStudies = [
    {
        id: 'cs1',
        image : '/img/icon1.png',
        title: 'Case Study 1',
        description: 'ChildCare Facilities Analysis',
        link: '/UseCases'
    },
    {
        id: 'cs2',
        image: '/img/icon2.png',
        title: 'Case Study 2',
        description: 'Projected Venue Growth',
        link: '/UseCases'
    },
    {
        id: 'cs3',
        image: '/img/icon3.png',
        title: 'Case Study 3',
        description: 'Data Science in Education',
        link: 'UseCases'
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
