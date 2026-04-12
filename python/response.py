from python.prompt import gen_prompt
from python.LLM import gpt, Gemini



async def LLM_response(inputData):
    
    prompt = await gen_prompt(inputData)
    response1 = await gpt(prompt)
    response2 = await Gemini(prompt)
    
    final_res = {
        "openai_response": response1,
        "gemini_response": response2
    }
    
    return final_res