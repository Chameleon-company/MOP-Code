import React, { useState, useEffect } from "react";
import axios from "axios";
import Card from 'react-bootstrap/Card'
import Col from 'react-bootstrap/Col' 
import Row from 'react-bootstrap/Row'
import { UseCase } from "@/app/types";
import { ChevronLeft, ChevronRight, FileText, ArrowLeft } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function NewPreviewComponent() {
    const [usecase, setUsecase] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedCaseStudy, setSelectedCaseStudy] = useState<UseCase | undefined>(undefined);

    const getUsecases = async () => {
        await axios
            .get("/api/usecases")
            .then((response) => {
                setUsecase(response.data.useCases);
                setLoading(true);
            })
            .catch((error) => {
                console.log(error);
            })
    };

    useEffect(() => {
        getUsecases();
    }, []);

    const removeMd = require('remove-markdown');

    const handleCaseStudyClick = (usecase: UseCase) => {
        setSelectedCaseStudy(usecase);
    };

    const handleBack = () => {
        setSelectedCaseStudy(undefined);
    };

    // selected preview
    if (selectedCaseStudy) {
        return (
        <div className="flex flex-col bg-gray-100 p-8">
            <button
                onClick={handleBack}
                className="flex items-center text-green-500 mb-4 hover:text-green-700 transition-colors duration-300"
            >
                <ArrowLeft size={24} className="mr-2" />
                Back
            </button>
            <div className="bg-white rounded-lg shadow-md p-6 flex-grow overflow-hidden">
                <h1 className="text-3xl font-bold mb-4">{selectedCaseStudy.name}</h1>
                <div className="text-lg">
                    <p><b>Authored by: </b>{selectedCaseStudy.auth}</p>
                    <br/>
                    <p><b>Duration: </b>{selectedCaseStudy.duration}</p>
                    <p><b>Level: </b>{selectedCaseStudy.level}</p>
                    <p><b>Pre-requisite Skills: </b>{selectedCaseStudy.skills}</p>
                    <br/>
                </div>
                <div className="text-xl">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {selectedCaseStudy.description.toString()}
                    </ReactMarkdown>
                </div>
            </div>
        </div>
        );
  }

    return (
        <>
        {/* cards preview */}
        <div className="w-full p-6">
            <div className="flex flex-wrap justify-left">
            {loading && usecase.map((useCases) => (
                <div className="flex flex-col cursor-pointer bg-white rounded-2xl shadow-md w-full m-4 overflow-hidden w-[400px]"
                onClick={() => handleCaseStudyClick(useCases)}>
                    <div className="p-7 text-xl"> 
                        <h1 className="font-extrabold text-gray-800 mb-2">{useCases.name}</h1>
                        <p className="text-slate-600 text-base leading-normal font-light mb-2">
                            {/* this removes the markdown loaded from the database description */}
                            {
                                removeMd(useCases.description).length <= 100 ? removeMd(useCases.description) : removeMd(useCases.description).substring(0, 100) + '...'
                            }
                        </p>
                        <div className="flex gap-3">
                            <div className="rounded-full bg-gray-200 py-1 px-4 text-gray-800">{useCases.skills}</div>
                        </div>
                    </div>
                </div>
            ))}
            </div>
        </div>
        </>
    )
}