# this is a free floating demonstration and not part of Tether.
# 

from openai import OpenAI

client = OpenAI()
import os

model_choice = 'o1' # 'gpt-4o' #'o3-mini' # 'gpt-4.5-preview' # 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)  # Updated client initialization

def ask_openai(question, model_choice, reasoning_effort='medium'):
    if model_choice == 'gpt-4o' or model_choice == 'gpt-4.5-preview':
        try:
            response = client.chat.completions.create(
                model=model_choice,  # gpt-4.5-preview, gpt-4o
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
    elif model_choice == 'o3-mini' or model_choice == 'o1':
        try:
            response = client.chat.completions.create(model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            reasoning_effort=reasoning_effort  # Options: 'low', 'medium', 'high')
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    user_question = input("Ask a question: ")
    answer = ask_openai(user_question,model_choice)
    print("\nOpenAI's response:\n", answer)

