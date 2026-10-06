from dotenv import load_dotenv
load_dotenv()
import time
import litellm
from litellm import completion

litellm.suppress_debug_info = True  # hides the "Give Feedback" noise

# Tried in order. If one is overloaded, we move to the next.
MODELS = [
    "gemini/gemini-flash-latest",
    "gemini/gemini-flash-lite-latest",
    "gemini/gemini-2.5-flash",
]


def load_info():
    with open("restaurant_info.txt", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT = f"""You are the friendly customer assistant for Ember & Spice Kitchen.
Answer ONLY using the restaurant information below.
If the answer is not in the information, say: "I'm not sure about that, let me get a team member to help you." Do not guess.
Never invent prices, hours, ingredients, or allergen details.
Keep answers short and friendly.
For allergy questions: say which listed items contain that allergen, say that our info only lists main ingredients and doesn't cover cross-contact in the kitchen, and suggest confirming with a team member by phone (555-0123) before ordering. Never say any item is completely safe.
RESTAURANT INFORMATION:
{load_info()}
"""


def ask(messages):
    for model in MODELS:
        for attempt in range(2):
            try:
                response = completion(model=model, messages=messages, max_tokens=2048)
                print(f"[answered by {model}]")
                return response.choices[0].message.content
            except Exception as e:
                print(f"[{model} attempt {attempt + 1} failed: {type(e).__name__}]")
                time.sleep(1)
    return None


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Ember & Spice assistant (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": user_input})
        reply = ask(messages)
        if reply is None:
            print("Bot: Sorry, I'm having trouble right now. Please try again in a moment.")
            messages.pop()  # forget the unanswered question
            continue
        messages.append({"role": "assistant", "content": reply})
        print("Bot:", reply)


if __name__ == "__main__":
    main()