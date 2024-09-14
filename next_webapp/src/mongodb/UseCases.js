const dbConnect = require('../../lib/mongodb');
const UseCase = require('./UseCase');

async function handleRequest(req, res) {
    await dbConnect();

    const { method, query } = req;
    const { tag, trimester, stats } = query;

    switch (method) {
        case 'GET':
            if (stats === 'true') {
                console.log('Called Stats method');
                // Prepare filter criteria
                const filter = {};
                if (tag) filter.tag = tag;
                if (trimester) filter.trimester = trimester;

                try {
                    const stats = await UseCase.aggregate([
                        {
                            $match: filter // Apply filters based on tag and trimester
                        },
                        {
                            $group: {
                                _id: {
                                    tag: "$tag",
                                    trimester: "$trimester"
                                },
                                totalPublishNumber: { $sum: { $toInt: "$publishNumber" } }, // Sum of publish numbers
                                averagePopularity: {
                                    $avg: {
                                        $toDouble: {
                                            $substr: ["$popularity", 0, {
                                                $subtract: [
                                                    { $strLenCP: "$popularity" },
                                                    1
                                                ]
                                            }]
                                        }
                                    }
                                }, // Average popularity
                                count: { $sum: 1 } // Count of use cases
                            }
                        },
                        {
                            $project: {
                                _id: 0,
                                tag: "$_id.tag",
                                trimester: "$_id.trimester",
                                totalPublishNumber: 1,
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
                console.log('Called Use Case Count by Trimester');
                // Handle count by trimester request
                try {
                    const trimesterCounts = await UseCase.aggregate([
                        {
                            $group: {
                                _id: "$trimester", // Group by trimester
                                count: { $sum: 1 } // Count the number of use cases in each trimester
                            }
                        },
                        {
                            $project: {
                                _id: 0,
                                trimester: "$_id",
                                count: 1
                            }
                        }
                    ]);
                    res.status(200).json({ success: true, data: trimesterCounts });
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
                res.status(400).json({ success: false, error: error.message });
            }
            break;
        default:
            res.setHeader('Allow', ['GET', 'POST']);
            res.status(405).end(`Method ${method} Not Allowed`);
    }
}

module.exports = handleRequest;