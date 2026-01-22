import os

from dotenv import load_dotenv
from google import genai


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def fix_spelling(self, query: str) -> str:
        prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
If you correct something, return just the corrected query.
Corrected:"""
        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text

    def rewrite_query(self, query: str) -> str:
        prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Return only the rewritten query, don't add anything before or after the query
Rewritten query:"""

        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text

    def expand_query(self, query: str) -> str:
        prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
Return only the expanded query, don't add anything before or after the query
"""

        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text


# import os
# from dotenv import load_dotenv
# from google import genai

# load_dotenv()
# api_key = os.environ.get("GEMINI_API_KEY")
# MODEL = "gemini-2.5-flash"

# client = genai.Client(api_key=api_key)

# prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

# response = client.models.generate_content(model=MODEL, contents=prompt)

# print(response.text)
# print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
# print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")
