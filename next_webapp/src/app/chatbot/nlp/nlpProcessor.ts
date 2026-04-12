import trainingData from "./rasa_training_data.json";

interface RasaExample {
  intent: string;
  examples: string;
}

interface RasaTrainingData {
  version: string;
  nlu: RasaExample[];
}

const rasaData: RasaTrainingData = trainingData as RasaTrainingData;

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export const processInput = (input: string): string => {
  const lowerInput = input.toLowerCase();
  let matchedIntent = "fallback";
  let bestMatchCount = 0;

  for (const intentData of rasaData.nlu) {
    const examples = intentData.examples
      .split("\n")
      .map(e => e.replace("-", "").trim().toLowerCase())
      .filter(Boolean);

    let matchCount = 0;
    for (const example of examples) {
      const regex = new RegExp(`\\b${escapeRegExp(example)}\\b`, "i");
      if (regex.test(lowerInput)) {
        matchCount++;
      }
    }
    if (matchCount > bestMatchCount) {
      bestMatchCount = matchCount;
      matchedIntent = intentData.intent;
    }
  }
  return matchedIntent;
};
