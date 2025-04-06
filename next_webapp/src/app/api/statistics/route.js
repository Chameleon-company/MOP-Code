// api/statistics.js

export default async function handler(req, res) {
  if (req.method === 'GET') {
    try {
      const { db } = await dbConnect();
      
      const useCaseCollection = db.collection('usecases');
      
      const trimesterStats = await useCaseCollection.aggregate([
        {
          $group: {
            _id: "$trimester",
            count: { $sum: 1 }
          }
        },
        {
          $sort: { _id: 1 }
        }
      ]).toArray();
      
      const formattedTrimesterStats = {
        labels: trimesterStats.map(item => `Trimester ${item._id}`),
        data: trimesterStats.map(item => item.count)
      };
      
      // Get tag statistics
      const tagStats = await useCaseCollection.aggregate([
        { $unwind: "$tags" },
        {
          $group: {
            _id: "$tags",
            count: { $sum: 1 }
          }
        },
        {
          $project: {
            tag: "$_id",
            publishNumber: "$count",
            _id: 0
          }
        },
        {
          $sort: { publishNumber: -1 }
        }
      ]).toArray();
      
      const totalTags = tagStats.reduce((sum, item) => sum + item.publishNumber, 0);
      
      const tagStatsWithPopularity = tagStats.map((item, index) => ({
        id: index + 1,
        tag: item.tag,
        publishNumber: item.publishNumber,
        popularity: `${((item.publishNumber / totalTags) * 100).toFixed(2)}%`,
        trimester: "2024 T1" 
      }));
      
   
      return res.status(200).json({
        trimesterStats: formattedTrimesterStats,
        tagStats: tagStatsWithPopularity
      });
      
    } catch (error) {
      console.error("Failed to fetch statistics:", error);
      return res.status(500).json({ error: "Failed to fetch statistics" });
    }
  } else {
    return res.status(405).json({ error: "Method not allowed" });
  }
}