import os

import litellm
from dotenv import load_dotenv

from src.components.article import Article

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL_NAME = "openrouter/meta-llama/llama-3.3-70b-instruct:free"

PROFILE_GENERATION_PROMPT_A = (
    "Below are the titles and abstracts of scientific papers I have authored. "
    "Based on this information, generate a concise first-person research "
    "profile that summarizes my main research interests and areas of expertise. "
    "The profile should be written in no more than three sentences and should "
    "clearly identify the key themes, research trends, and topics I focus on. "
    "The purpose of this profile is to support a scientific literature "
    "recommendation system, so it should accurately reflect my research focus "
    "to help match me with relevant publications. \n\n"
    "This is the list of my publications:"
    "\n\n{}\n\n"
    "Write the profile in first person. "
    "Return nothing but the generated profile."
)

PROFILE_GENERATION_PROMPT_B = (
    "Below are the titles and abstracts of scientific papers I have authored. "
    "Based on this information, generate a concise description of my research "
    "interests, characterizing the key topics and areas of expertise."
    "This is the list of my publications:"
    "\n\n{}\n\n"
    "Write the profile in first person. "
    "Return nothing but the generated profile."
)


class ProfileGenerator:
    def __init__(self):
        """Initializes the ProfileGenerator.

        Args:
            model_name: Model to use.
        """
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

    def generate_profile(
        self,
        articles: list[Article],
        base_prompt: str = PROFILE_GENERATION_PROMPT_A,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> str:
        """Generates a natural language profile from a list of papers.

        Args:
            articles: List of articles.
            base_prompt: Base prompt to use for profile generation.

        Returns:
            str: Generated natural language profile.
        """
        prompt = self._create_prompt(articles, base_prompt)
        profile = self._query_llm(prompt, model_name)
        return profile

    def _create_prompt(self, articles: list[Article], base_prompt: str) -> str:
        """Creates LLM prompt based on a list of papers.

        Args:
            articles: List of articles.
            base_prompt: Base prompt to use for profile generation.

        Returns:
            str: Prompt string including nl_profile_input article titles and
            abstracts.
        """
        paper_descriptions = "\n\n".join(
            [
                f"Title: {article.title}\nAbstract: {article.abstract}"
                for article in articles
            ]
        )

        return base_prompt.format(paper_descriptions)

    def _query_llm(self, prompt: str, model_name: str) -> str:
        """Sends the prompt to the selected LLM and returns the output.

        Args:
            prompt: Input prompt for the LLM.

        Returns:
            str: LLM-generated output.
        """

        response = litellm.completion(
            model=model_name,
            api_key=os.environ["OPENROUTER_API_KEY"],
            api_base="https://openrouter.ai/api/v1",
            messages=[{"role": "user", "content": prompt}],
        )

        return response["choices"][0]["message"]["content"]
