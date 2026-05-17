const dbConnect = require('../../lib/mongodb');
const handleRequest = require('./UseCases');
const UseCase = require('./UseCase');
async function main() {
    await dbConnect();

    console.log('Inserting dummy data...');
    await UseCase.deleteMany({}); // Clear existing data

    const dummyData = [
        { tag: 'Frontend', semester: '2024-T1', popularity: 75 },
        { tag: 'Backend', semester: '2024-T2', popularity: 60 },
        { tag: 'Frontend', semester: '2024-T3', popularity: 80 },
        { tag: 'Frontend', semester: '2024-T1', popularity: 55 },
        { tag: 'Backend', semester: '2024-T3', popularity: 70 }
    ];

    await UseCase.insertMany(dummyData);
    console.log('Dummy data inserted.');

    // Mock req and res objects
    const req = {
        method: 'GET',
        query: { stats: 'true' }
    };

    const res = {
        status: (statusCode) => {
            res.statusCode = statusCode;
            return res;
        },
        json: (data) => {
            console.log('Response:', data);
        },
        end: (message) => {
            console.log('Response ended:', message);
        },
        statusCode: null
    };

    try {
        console.log('Fetching use case statistics...');
        await handleRequest(req, res);
        console.log('Operation completed.');
    } catch (error) {
        console.error('Error executing request:', error.message);
    }
}

// Execute the main function if this module is run directly
if (require.main === module) {
    main().catch(error => {
        console.error('Unhandled error:', error.message);
        process.exit(1);
    });
}