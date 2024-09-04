const { MongoClient } = require('mongodb');
const fs = require('fs');

const uri = 'mongodb+srv://mop:mop@mop.tzmys.mongodb.net/usecasesdb';

const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

const filePath = 'E:/Deakin/Lectures/2024/Trimester 2 2024/SIT782/use cases/Bus stop and pedestrian Analysis.ipynb';

const notebookContent = JSON.parse(fs.readFileSync(filePath, 'utf8'));

async function run() {
  try {
    await client.connect();
    console.log('Connected to MongoDB');

    const database = client.db('usecasesdb');
    const collection = database.collection('usecases');

    const result = await collection.insertOne(notebookContent);
    console.log('Notebook uploaded:', result.insertedId);
  } catch (err) {
    console.error('Error uploading notebook:', err);
  } finally {
    await client.close();
  }
}

run().catch(console.dir);
