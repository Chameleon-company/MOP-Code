async def gen_prompt(data):
    
    prompt = f"""
    You are an urban infrastructure assistant.

    Generate a short maintenance report based on the following data:

    Location: {data['location']}
    Number of faulty streetlights: {data['faulty_lights']}
    Issues detected: {data['issues']}
    Priority level: {data['priority']}

    Write a clear and professional report in 2-3 sentences.
    """
    
    return prompt