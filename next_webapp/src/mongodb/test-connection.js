// pages/api/test-connection.js
const dbConnect = require('../../lib/mongodb');
const Item = require('../../models/Item');

export default async function handler(req, res) {
  await dbConnect();

  try {
    const items = await Item.find({});
    res.status(200).json({ success: true, data: items });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
}
