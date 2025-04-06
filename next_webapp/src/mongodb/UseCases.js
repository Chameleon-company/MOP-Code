const dbConnect = require('../../lib/mongodb');
const UseCase = require('./UseCase');

async function handleRequest(req, res) {
    await dbConnect();

    const { method } = req;

    switch (method) {
        case 'GET':
            if (req.query.stats === 'true') {
                console.log('Called Stats method');
                // Handle statistics request
                try {
                    const stats = await UseCase.aggregate([
                        {
                            $group: {
                                _id: {
                                    tag: "$tag",
                                    semester: "$semester"
                                },
                                averagePopularity: { $avg: "$popularity" },
                                count: { $sum: 1 }
                            }
                        },
                        {
                            $project: {
                                _id: 0,
                                tag: "$_id.tag",
                                semester: "$_id.semester",
                                averagePopularity: { $round: ["$averagePopularity", 2] }, // Optional: round to 2 decimal places
                                count: 1
                            }
                        }
                    ]);
                    res.status(200).json({ success: true, data: stats });
                } catch (error) {
                    res.status(400).json({ success: false, error: error.message });
                }
            } else {
                // Handle normal GET request
                try {
                    const useCases = await UseCase.find({});
                    res.status(200).json({ success: true, data: useCases });
                } catch (error) {
                    res.status(400).json({ success: false, error: error.message });
                }
            }
            break;
        case 'POST':
            try {
                const useCase = await UseCase.create(req.body);
                res.status(201).json({ success: true, data: useCase });
            } catch (error) {
                console.log('haritha error!');
                res.status(400).json({ success: false, error: error.message });
            }
            break;
        default:
            res.setHeader('Allow', ['GET', 'POST']);
            res.status(405).end(`Method ${method} Not Allowed`);
    }
}

module.exports = handleRequest;