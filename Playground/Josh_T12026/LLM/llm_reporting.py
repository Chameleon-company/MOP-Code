import os
import json

from openai import AsyncOpenAI

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def gen_prompt(data):

    analysis = data["analysis"]
    
    prompt = f"""
    You are an AI-powered streetlight monitoring assistant.

    An uploaded streetlight image is provided together with ML detection results.

    ML Detection Results:
    - Total Streetlights: {analysis["streetlight_count"]}
    - ON Lights: {analysis["on"]}
    - DIM Lights: {analysis["dim"]}
    - OFF Lights: {analysis["off"]}
    - Detection Details: {analysis["details"]}

    Instructions:
    - Analyze the uploaded image together with the ML output.
    - Describe the overall streetlight condition naturally.
    - Mention operational, dim, and faulty streetlights.
    - Highlight maintenance concerns if necessary.
    - Keep the response concise and professional.
    - Do not mention Base64 data.

    Return ONLY valid JSON:

    {{
        "output": "your report here"
    }}
    """

    return prompt


async def gpt(prompt, base64_image):

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    completion = await client.chat.completions.create(
        model="gpt-4.1-mini",

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],

        max_tokens=800,
        temperature=0
    )

    response = completion.choices[0].message.content

    return json.loads(response)


async def llm_reporting(data):

    base64_image = data["uploaded_img"]

    prompt = await gen_prompt(data)

    response = await gpt(prompt, base64_image)

    return response