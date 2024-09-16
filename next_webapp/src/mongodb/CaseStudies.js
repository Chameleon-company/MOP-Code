const dbConnect = require('../../lib/mongodb');
const CaseStudy = require('./CaseStudy');

async function handleRequest(req, res) {
    await dbConnect();

    const { method } = req;

    switch (method) {
        case 'GET':
            console.log('Called all studies');
            try {
                // Retrieve all case studies without the _id field
                const caseStudies = await CaseStudy.find({})
                    .select('-_id'); // Exclude the _id field

                res.status(200).json({ success: true, data: caseStudies });
            } catch (error) {
                res.status(400).json({ success: false, error: error.message });
            }
            break;
        case 'POST':
            try {
                const useCase = await CaseStudy.create(req.body);
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