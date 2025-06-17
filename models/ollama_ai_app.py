import streamlit as st
import gc
from ollama import Client

def clear_memory():
    gc.collect()

def is_valid_recipe(text):
    lines = text.strip().lower().splitlines()
    if not lines:
        return False

    bad_keywords = [
        "follow me", "on my blog", "thanks for stopping by", "i will", "check out",
        "i'll post", "leave a comment", "subscribe", "don't forget to", "share this post",
        "visit my", "subscribe for updates", "like and share", "this recipe is", "my favorite",
        "click here", "for more info", "be sure to", "join me", "sign up", "exclusive content",
        "click below", "read more", "helpful tips", "you can also", "this is my", "my journey",
        "download the", "free download", "download now", "see the recipe", "follow on",
        "follow on instagram", "check out my", "recipe link", "affiliate link", "thank you",
        "let me know", "here's the link", "free gift", "freebie", "limited time offer", "don't miss",
        "watch now", "learn more", "join us", "get started", "next time", "see you soon",
        "learn how to", "join me on", "explore more", "new post", "save time", "you can find", "start now"
    ]

    for keyword in bad_keywords:
        if any(keyword in line for line in lines):
            return False

    has_bullets = sum(1 for line in lines if line.strip().startswith(("-", "•"))) >= 10
    return has_bullets

def generate_recipes(ingredients):
    if len(ingredients) < 3:
        st.warning("Please enter at least 3 ingredients before generating recipes.")
        return "Insufficient ingredients."

    prompt = (
        f"You are a professional chef. Create 3 original recipes using ONLY these ingredients: {', '.join(ingredients)}.\n\n"
        "STRICT RULES (you must follow these exactly):\n"
        "- Use ONLY the ingredients listed. Absolutely do NOT assume, invent, or add anything else.\n"
        "- If an ingredient is not in the list, do NOT mention it — not even as an optional suggestion.\n"
        "- Do NOT include phrases like 'let's assume', 'feel free to add', or 'optional'.\n\n"
        "Format each recipe like this:\n"
        "### 1. Recipe Title\n"
        "**Ingredients:**\n"
        "- item (quantity)\n"
        "**Instructions:**\n"
        "1. Step one...\n"
        "2. Step two...\n\n"
        "Write clearly and professionally, and keep your output strictly focused on the provided ingredients only."
    )


    ollama = Client()
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = ollama.chat(
                model="llama3.2:latest",
                messages=[{"role": "user", "content": prompt}]
            )
            text = response['message']['content']
            cleaned = "\n".join(line for line in text.splitlines() if not line.strip().isdigit())

            if is_valid_recipe(cleaned):
                return cleaned.strip()
            else:
                st.warning(f"⚠️ Attempt {attempt}/{max_attempts} failed validation. Retrying...")
        except Exception as e:
            st.error(f"❌ Error on attempt {attempt}: {str(e)}")
            return f"Error: {str(e)}"

    return "Failed to generate valid recipes after multiple attempts. Please try different ingredients or refresh."

def main():
    st.set_page_config(page_title="AI Recipe Generator", layout="wide")
    st.title("🍳 AI Recipe Generator")
    st.caption("Enter ingredients and get creative recipes powered by an AI chef.")

    if "ingredients" not in st.session_state:
        st.session_state.ingredients = []

    with st.form(key="ingredient_form", clear_on_submit=True):
        ingredient_input = st.text_input(
            "Enter an ingredient (specify how much you have of this ingredient to get even better results):",
            key="ingredient_input"
        )
        submitted = st.form_submit_button("➕ Add Ingredient")

    if submitted:
        ingredient = ingredient_input.strip().lower()
        if not ingredient:
            st.error("Please enter ingredients first.")
        elif ingredient in st.session_state.ingredients:
            st.warning(f"'{ingredient}' is already in the list.")
        else:
            st.session_state.ingredients.append(ingredient)
            st.success(f"Added: {ingredient}")

    if st.button("🗑️ Clear All"):
        st.session_state.ingredients = []
        st.success("All ingredients cleared.")
        st.rerun()

    if st.session_state.ingredients:
        st.markdown("### 🧾 Your Ingredients:")
        st.write(", ".join(st.session_state.ingredients))

    if len(st.session_state.ingredients) < 3:
        st.info("Please enter at least 3 ingredients to generate recipes.")
    else:
        if st.button("👨‍🍳 Generate Recipes"):
            recipes = generate_recipes(st.session_state.ingredients)
            st.markdown("### 🧑‍🍳 Your Generated Recipes:")
            st.write(recipes)

    clear_memory()

if __name__ == "__main__":
    main()
