const dbConnect = require('../../lib/mongodb');
const handleRequest = require('./CaseStudies');
const CaseStudy = require('./CaseStudy');

async function main() {
    await dbConnect();

    console.log('Inserting dummy data...');
    await CaseStudy.deleteMany({}); // Clear existing data

    // Dummy data provided
    const CaseStudies = [
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

    await CaseStudy.insertMany(CaseStudies);
    console.log('Dummy data inserted.');

    //Fetch all use cases
    const reqCount = {
        method: 'GET',
        query: {}
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