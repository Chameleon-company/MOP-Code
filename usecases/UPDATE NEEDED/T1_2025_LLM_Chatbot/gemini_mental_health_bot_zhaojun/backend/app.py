from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
from bert_score import score  # Corrected import

load_dotenv()  # Load environment variables from .env file
nltk.download('punkt')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the API key from the environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

conversation_history = []


def get_initial_response():
    """
    Returns a personalized initial response.
    """
    return "Hello! How are you feeling today? I'm here to listen and provide support."


def chat_with_gemini(user_input, history):
    """
    Sends user input to the Gemini Pro model and gets the response.

    Args:
        user_input (str): The user's message.
        history (list):  The conversation history.

    Returns:
        str: The chatbot's response.
    """
    try:
        # Use the global model variable
        global model
        prompt = f"""You are a supportive mental health assistant.
        A user is feeling anxious and needs guidance.  Offer practical, empathetic advice.
        Here is the conversation history:
        {history}
        User: {user_input}
        """
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "top_k": 50, "top_p": 0.9, "max_output_tokens": 200},
        )
        if response and response.text:
            return response.text
        else:
            return "I'm having trouble processing your request right now. Please try again later."
    except Exception as e:
        print(f"Error in chat_with_gemini: {e}")
        return f"An error occurred: {e}"


def process_message(message, history):
    """
    Processes the user's message and generates a response.

    Args:
        message (str): The user's message.
        history (list): The conversation history.

    Returns:
        str: The chatbot's response.
    """
    response = chat_with_gemini(message, history)
    history.append({"user": message, "bot": response})
    return response


@app.route("/chat", methods=["POST"])  # Only accept POST requests for /chat
def chat():
    """
    Handles incoming chat messages. Supports only POST requests.
    """
    try:
        data = request.get_json()
        message = data["message"]
        global conversation_history  # Access the global variable
        response = process_message(message, conversation_history)
        return jsonify({"response": response, "history": conversation_history})
    except KeyError as e:
        print(f"Missing key in request: {e}")
        return jsonify({"error": "Invalid request format. 'message' key is required."}), 400
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


@app.route("/start_chat", methods=["GET"])
def start_chat():
    """
    Clears the chat history and returns the initial response.
    """
    global conversation_history
    conversation_history = []
    initial_response = get_initial_response()
    return jsonify({"response": initial_response, "history": conversation_history})


def evaluate_model(predictions, references):
    """
    Evaluates the model's performance using ROUGE, BLEU, and BERTScore.

    Args:
        predictions (list): A list of model-generated responses.
        references (list): A list of corresponding ground truth responses.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": [], "bleu": [], "bertscore": []}

    for pred, ref in zip(predictions, references):
        rouge_scores = scorer.score(ref, pred)
        scores["rouge1"].append(rouge_scores["rouge1"].fmeasure)
        scores["rouge2"].append(rouge_scores["rouge2"].fmeasure)
        scores["rougeL"].append(rouge_scores["rougeL"].fmeasure)
        try:
            scores["bleu"].append(sentence_bleu([ref.split()], pred.split()))
        except Exception as e:
            print(f"Error calculating BLEU: {e}, pred: {pred}, ref: {ref}")
            scores["bleu"].append(0)  # Append 0 if there's an error

    # BERTScore
    P, R, F1 = score(predictions, references, lang="en", model_type="bert-base-uncased")
    scores["bertscore"] = F1.tolist()  # Convert to list

    return {k: sum(v) / len(v) if v else 0 for k, v in scores.items()}  # Return averages


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Evaluates the chatbot's responses. This is a simplified example. In a real
    application, you'd load test data from a file or database.
    """
    try:
        data = request.get_json()
        test_questions = data["questions"]
        test_answers = data["answers"]

        model_responses = [chat_with_gemini(q, conversation_history) for q in test_questions]  # Pass the history
        results = evaluate_model(model_responses, test_answers)
        return jsonify(results)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return jsonify({"error": "An error occurred during evaluation."}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_ENV") == "development"
    app.run(debug=debug_mode, port=5001)  # Changed port to 5001
