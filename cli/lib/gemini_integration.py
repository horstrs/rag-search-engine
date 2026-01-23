import os
import json

from dotenv import load_dotenv
from google import genai


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.rerank_precision = 3

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

    def individual_rerank(self, query: str, movie: dict) -> str:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {movie.get("title", "")} - {movie.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.
The score should be a float with up to {self.rerank_precision} decimal points of precision

Score:"""
        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return float(response.text)

    def batch_rerank(self, query: str, doc_list: list[str]) -> any:
        prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        return json.loads(response.text)

    def evaluate_results(self, query: str, formatted_results: list[str]) -> any:
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        return json.loads(response.text)

    def rag(self, query: str, docs: list[str]) -> str:
        prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text

    def summarize(self, query: str, docs: list[str]) -> str:
        prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs}
Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
"""

        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text

    def citations(self, query: str, docs: list[str]) -> str:
        prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- Add the citations at the end of your answer in a list. Example:
   -- 1. <document_1_title>
   -- 2. <document_2_title>
- Don't skip any number when creating the citations
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text
