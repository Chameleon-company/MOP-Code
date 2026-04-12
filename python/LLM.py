import os
from openai import AsyncOpenAI

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()



# Test with OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def gpt(prompt):
    
    chat_prompt = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    completion = await gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_prompt,
        max_tokens=100,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    response = completion.choices[0].message.content
    return response


# Test with Gemini
GEMINI_KEY = os.getenv("GEMINI_KEY")
gemini_client = genai.Client(api_key=GEMINI_KEY)


async def Gemini(Prompt):

    chat_prompt = [types.Part.from_text(text=Prompt)]

    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=chat_prompt
    )

    return response.text
