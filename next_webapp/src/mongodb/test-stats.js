const dbConnect = require('../../lib/mongodb');
const handleRequest = require('./UseCases');
const UseCase = require('./UseCase');

async function main() {
    await dbConnect();

    console.log('Inserting dummy data...');
    await UseCase.deleteMany({}); // Clear existing data

    // Dummy data provided
    const caseStudies = [
        { tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "1" },
        { tag: "Environment and Sustainability", publishNumber: "5", popularity: "40%", trimester: "2" },
        { tag: "Business and activity", publishNumber: "8", popularity: "90%", trimester: "3" },
        { tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "2" },
        { tag: "Environment and Sustainability", publishNumber: "5", popularity: "90%", trimester: "3" },
        { tag: "Business and activity", publishNumber: "8", popularity: "70%", trimester: "1" },
        { tag: "Safety and Well-being", publishNumber: "4", popularity: "11%", trimester: "2" },
        { tag: "Environment and Sustainability", publishNumber: "5", popularity: "20%", trimester: "2" },
        { tag: "Business and activity", publishNumber: "8", popularity: "60%", trimester: "1" },
    ];

    await UseCase.insertMany(caseStudies);
    console.log('Dummy data inserted.');

    // Test case 1: Fetch use case statistics with filters
    const reqStats = {
        method: 'GET',
        query: { stats: 'true', tag: 'Environment and Sustainability', trimester: '2' }
    };

    const resStats = {
        status: (statusCode) => {
            resStats.statusCode = statusCode;
            return resStats;
        },
        json: (data) => {
            console.log('Response for stats request:', data);
        },
        end: (message) => {
            console.log('Response ended:', message);
        },
        statusCode: null
    };

    try {
        console.log('Fetching use case statistics...');
        await handleRequest(reqStats, resStats);
    } catch (error) {
        console.error('Error executing stats request:', error.message);
    }

    // Test case 2: Fetch use case count by trimester
    const reqCount = {
        method: 'GET',
        query: {} // No filters for counting by trimester
    };

    const resCount = {
        status: (statusCode) => {
            resCount.statusCode = statusCode;
            return resCount;
        },
        json: (data) => {
            console.log('Response for count request:', data);
        },
        end: (message) => {
            console.log('Response ended:', message);
        },
        statusCode: null
    };

    try {
        console.log('Fetching use case count by trimester...');
        await handleRequest(reqCount, resCount);
    } catch (error) {
        console.error('Error executing count request:', error.message);
    }

    console.log('Operation completed.');
}

// Execute the main function if this module is run directly
if (require.main === module) {
    main().catch(error => {
        console.error('Unhandled error:', error.message);
        process.exit(1);
    });
}