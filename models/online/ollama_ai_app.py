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

    has_bullets = any(line.strip().startswith(("-", "•")) for line in lines)
    return has_bullets

def generate_recipes(ingredients):
    if len(ingredients) < 3:
        st.warning("Please enter at least 3 ingredients before generating recipes.")
        return "Insufficient ingredients."

    prompt = (
        f"As a professional chef, create 3 unique recipes using ONLY these ingredients: {', '.join(ingredients)}.\n\n"
        "For each recipe, provide:\n"
        "- A title that reflects the ingredients used\n"
        "- A list of ingredients with quantities\n"
        "- At least 5 clear cooking instructions\n\n"
        "Ensure that the recipe title accurately represents the dish and doesn't include ingredients that aren't in the list. "
        "Make sure the recipes are original and don't include common phrases or promotional content.\n"
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

def add_ingredient_from_enter():
    ingredient = st.session_state.ingredient_input.strip().lower()
    if not ingredient:
        st.error("Please enter ingredients first.")
        return
    if 'ingredients' not in st.session_state:
        st.session_state.ingredients = []
    if ingredient in st.session_state.ingredients:
        st.warning(f"'{ingredient}' is already in the list.")
    else:
        st.session_state.ingredients.append(ingredient)
        st.success(f"Added: {ingredient}")
    st.session_state.ingredient_input = ""  # Safe in on_change context

def main():
    st.set_page_config(page_title="AI Recipe Generator", layout="wide")
    st.title("🍳 AI Recipe Generator")
    st.caption("Enter ingredients and get creative recipes powered by an AI chef.")

    if 'ingredients' not in st.session_state:
        st.session_state.ingredients = []

    st.text_input(
        "Enter an ingredient:",
        key="ingredient_input",
        on_change=add_ingredient_from_enter
    )

    if st.button("➕ Add Ingredient"):
        ingredient = st.session_state.ingredient_input.strip().lower()
        if not ingredient:
            st.error("Please enter ingredients first.")
        elif ingredient in st.session_state.ingredients:
            st.warning(f"'{ingredient}' is already in the list.")
        else:
            st.session_state.ingredients.append(ingredient)
            st.success(f"Added: {ingredient}")
            st.session_state.ingredient_input = ""

    if st.button("🗑️ Clear All"):
        st.session_state.ingredients = []
        st.success("All ingredients cleared.")

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
