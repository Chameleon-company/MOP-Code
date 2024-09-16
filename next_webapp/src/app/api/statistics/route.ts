import handleRequest from '../../../mongodb/CaseStudies';

export async function GET(request: Request) {
    try {
        const reqCount = {
            method: 'GET',
            query: {}
        };

        const resStats = {
            status({statusCode}: { statusCode: any }) {
                this.statusCode = statusCode;
                return this;
            },
            json({data}: { data: any }) {
                this.data = data
                console.log('Response for count request:', data);
            },
            end({message}: { message: any }) {
                this.message = message
                console.log('Response ended:', message);
            },
            statusCode: 400,
            data: {},
            message: ""
        };

        await handleRequest(reqCount, resStats);
        const statusCode = resStats.statusCode | 200;
        const data =  resStats.data ;

        // Returning the response data
        return new Response(JSON.stringify(data), {
            status: statusCode,
            headers: { 'Content-Type': 'application/json' },
        });
    } catch (error) {
        console.error('Error handling request:', error);
        return new Response(JSON.stringify({ message: 'Failed to fetch statistics' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
        });
    }
}