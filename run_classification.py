# File: run_classification.py
import os
import json
import logging
from tqdm import tqdm
from typing import Any, Optional, List, Dict
import time
from dotenv import load_dotenv
import litellm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_not_exception_type
from pydantic import BaseModel, Field, field_validator
from litellm.types.utils import ModelResponse
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

# --- Helper Classes ---
MODEL_ALIASES = {
    "gpt4o": "gpt-4o",
}

class APIModelConfig(BaseModel):
    """Configuration for the API model."""
    model_name: str = Field(default="gpt4o")
    temperature: float = Field(default=0.0)  # Zero temp for consistent classification
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=4096)  # Fixed: GPT-4o max is 4096
    host_url: Optional[str] = Field(default=None)
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v in MODEL_ALIASES:
            return MODEL_ALIASES[v]
        return v

class APIStats(BaseModel):
    """Statistics for the API model."""
    cost: float = 0.0
    tokens_received: int = 0
    tokens_sent: int = 0
    api_calls: int = 0
    
    def __add__(self, other: "APIStats") -> "APIStats":
        if not isinstance(other, APIStats):
            raise TypeError("APIStats objects can only be added to other APIStats objects")
        return APIStats(**{field: getattr(self, field) + getattr(other, field) for field in self.model_fields})

# --- Main LiteLLM Wrapper Class ---
class LiteLLMModel:
    """A wrapper for the LiteLLM API."""
    
    def __init__(self, config: APIModelConfig, name: str, logger: logging.Logger) -> None:
        self.name = name
        self.config = config
        self.stats = APIStats()
        self.logger = logger
        self._setup_client()
    
    def _setup_client(self) -> None:
        self.model_name = self.config.model_name
        self.model_max_input_tokens = litellm.model_cost.get(self.model_name, {}).get("max_input_tokens")
        self.model_max_output_tokens = litellm.model_cost.get(self.model_name, {}).get("max_output_tokens")
        self.lm_provider = litellm.model_cost.get(self.model_name, {}).get("litellm_provider")
        if self.lm_provider is None or self.config.host_url is not None:
            self.logger.warning(f"Using custom host URL: {self.config.host_url}. Cost management may not be available.")
    
    def update_stats(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> None:
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.cost += cost
        self.stats.api_calls += 1
    
    @retry(
        wait=wait_random_exponential(min=1, max=60),  # More reasonable retry timing
        reraise=True,
        stop=stop_after_attempt(10),  # More attempts for rate limits
        retry=retry_if_not_exception_type(
            (NotFoundError, PermissionDeniedError, ContextWindowExceededError, AuthenticationError, BadRequestError)
        ),
    )
    def query(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
        input_tokens = litellm.utils.token_counter(messages=messages, model=self.model_name)
        extra_args = {}
        if self.config.host_url:
            extra_args["api_base"] = self.config.host_url
        
        completion_kwargs = self.config.completion_kwargs.copy()
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.config.max_tokens
        
        if tools is not None:
            completion_kwargs["tools"] = tools
        
        try:
            response: ModelResponse = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                stream=False,
                drop_params=True,
                **completion_kwargs,
                **extra_args,
            )
        except RateLimitError as e:
            self.logger.warning(f"Rate limit hit, retrying: {e}")
            raise
        except Exception:
            self.logger.exception(f"Error querying {self.model_name}")
            raise
        
        output_content = response.choices[0].message.content or ""
        output_tokens = response.usage.completion_tokens or litellm.utils.token_counter(text=output_content, model=self.model_name)
        cost = litellm.cost_calculator.completion_cost(response) or 0.0
        
        self.update_stats(input_tokens, output_tokens, cost)
        self.logger.info(
            f"API Call {self.stats.api_calls}: "
            f"Input Tokens={input_tokens:,}, Output Tokens={output_tokens:,}, Cost=${cost:.4f}. "
            f"Total Cost=${self.stats.cost:.2f}"
        )
        return response

# --- Application Logic ---
def get_classification_prompt() -> str:
    """Returns the detailed classification prompt."""
    return """Overview
You will classify individual texts (posts or replies) for hate speech and counterspeech. Each text is paired with a unique Twitter/X ID.
Your job: return the labels for each ID using the instructions detailed in the following two parts. Do not repeat the original text.

Part 1 - Hate Speech Classification
Fields to Add
1. Hate_speech:
    1. 1 if the text contains hate speech, 0 otherwise.
    2. Hate speech = any statement that attacks, demeans, or incites harm against a person or group based on one or more protected characteristics (race, ethnicity, religion, nationality, gender, sexual orientation, disability, etc.).
2. Hate_speech_type:
    1. "explicit" → overt, direct use of slurs, severe derogatory language, or clear calls for violence.
        1. Examples:
            1. "Gas all the Jews -- it's the only solution." (call for genocide)
            2. "Kill every single one of those goat-fucking refugees, they don't deserve to live." (incitement to mass violence)
            3. "Lynch that nigger before he infects the neighborhood." (explicit racial violence)
            4. "Fags aren't even human. they should be wiped out." (dehumanization + extermination)
    2. "implicit" → indirect, coded, sarcastic, or requiring context to detect -- often euphemisms, plausible deniability or dog whistles.
        1. Examples:
            1. "What do you say to a black guy in a three piece suit? Will the defendant please rise"
            2. "What part of a vegetable cant you eat? The wheelchair"
            3. "you want to hear a joke? Women 's rights"
            4. "Why should you never marry a jewish woman ? When dinner time comes, they 're too scared to get near the oven"
            5. "Whats the only positive thing about Ethiopia? The HIV tests"
    3. Hate_speech_severity:
        1. "low" → Stereotypes, generalizations, or mildly demeaning remarks with low likelihood of direct harm.
            1. Examples:
                1. "Women are too emotional to lead."
                2. "Old people can't understand technology."
        2. "medium" → Targeted insults, harassment, or demeaning language likely to reinforce discrimination but without direct calls for violence.
            1. Examples:
                1. "Send all the immigrants home, they don't belong here."
                2. "Gays are disgusting and ruin the moral fabric of society."
        3. "high" → Explicit calls for violence, genocide, or severe harassment likely to incite immediate harm.
            1. Examples:
                1. "Burn every mosque to the ground."
                2. "Every day when I see those fucking pride parades I want to beat up those faggots until I knock some sense into them"
                3. "ENOUGH IS ENOUGH. We NEED to rid this country of those godforsaken Mexicans. We NEED to do whatever it takes"
Part 2 - Counterspeech Classification
Fields to Add
1. Counterspeech:
    1. 1 if the text directly addresses hate speech with the aim of reducing its impact, 0 otherwise
2. Counterspeech_type:
    1. If counterspeech = 1, choose the relevant categories out of the following (else null). 

Category
Definition
Example

"presenting facts"
Correcting false or misleading statements with evidence.
"Actually, refugee crime rates are lower than native-born citizens."

"pointing out hypocrisy or contradictions"
Highlighting inconsistencies in the hate speaker's logic.
"You preach family values but cheat on your spouse -- maybe rethink judging others."

"warning of consequences"
Pointing out possible legal, social, or personal repercussions.
"Inciting violence online can get you charged with a felony."

"affiliation"
Establishing shared identity or solidarity with the target or both groups.
"I'm a veteran, and I fought for everyone's freedom -- including Muslims."

"denouncing hate speech"
Explicitly condemning the hateful message.
"This is blatant racism and it's disgusting."

"humor"
Using jokes, irony, or sarcasm to undermine hate.
"Wow, you've unlocked the 'Medieval Thinking' achievement badge."

"empathy/positive tone"
Expressing compassion, kindness, or unity.
"We're all trying to live decent lives -- hate helps no one."

"hostile language"
Responding to hate with insults or profanity while countering the message.
"Only an idiot would believe that garbage."

3. dominant_counterspeech_type:
   - If there is no counterspeech present, put null.
   - If there is one type of counterspeech present, put that type.
   - If there are multiple types present, choose the most salient and emphasized form. 

Never output original text, never add explanations, only return fields as instructed.

Example Input
{
   ID: 12345
   Text: We all know who controls the banks.
}
{
   ID: 67890
   Text: This is an antisemitic trope -- Jews do not control banking.
}
{
   ID: 11111
   Text: Exactly, they're behind everything.
}

Output Format:

IMPORTANT: Return your response as a valid JSON array containing objects for each ID. Each object must have these exact fields:
- "id": (the ID number)
- "hate_speech": (0 or 1)
- "hate_speech_type": ("explicit", "implicit", or null)
- "hate_speech_severity": ("low", "medium", "high", or null)
- "counterspeech": (0 or 1)
- "counterspeech_type": (one of the types listed above or null)
- "dominant_counterspeech_type": (same as counterspeech_type if only one, or the dominant one if multiple, or null)

Return ONLY the JSON array, no other text, no markdown formatting."""

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSON Lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if 'ID' not in item or 'Text' not in item:
                    logging.warning(f"Line {line_num}: Missing ID or Text field, skipping")
                    continue
                data.append(item)
            except json.JSONDecodeError as e:
                logging.warning(f"Line {line_num}: Malformed JSON - {e}")
    return data

def format_batch_for_prompt(batch: List[Dict[str, Any]]) -> str:
    """Formats a batch of data into a single string for the user prompt."""
    formatted_items = []
    for item in batch:
        formatted_items.append(f'{{\n  "ID": {item["ID"]},\n  "Text": "{item["Text"]}"\n}}')
    return "\n".join(formatted_items)

def parse_model_response(content: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse the model's response and extract classifications."""
    results = []
    
    try:
        # Clean up the content - remove any markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        parsed = json.loads(content)
        
        if isinstance(parsed, list):
            # Create a mapping for validation
            batch_ids = {item["ID"] for item in batch}
            
            for item in parsed:
                if isinstance(item, dict) and 'id' in item:
                    # Ensure all fields are present with correct names
                    result = {
                        "ID": item.get("id"),
                        "hate_speech": item.get("hate_speech", 0),
                        "hate_speech_type": item.get("hate_speech_type"),
                        "hate_speech_severity": item.get("hate_speech_severity"),
                        "counterspeech": item.get("counterspeech", 0),
                        "counterspeech_type": item.get("counterspeech_type"),
                        "dominant_counterspeech_type": item.get("dominant_counterspeech_type")
                    }
                    
                    # Validate the ID is in our batch
                    if result["ID"] in batch_ids:
                        results.append(result)
                    else:
                        logging.warning(f"Received ID {result['ID']} not in batch")
        else:
            logging.error("Model response is not a JSON array")
            
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse model response as JSON: {e}")
        logging.debug(f"Response content: {content[:500]}...")
        
        # Fallback: create placeholder results for the batch
        for item in batch:
            results.append({
                "ID": item["ID"],
                "hate_speech": -1,  # -1 indicates parsing error
                "hate_speech_type": "error",
                "hate_speech_severity": "error",
                "counterspeech": -1,
                "counterspeech_type": "error",
                "dominant_counterspeech_type": "error"
            })
    
    return results

def main():
    """Main function to run the classification task."""
    load_dotenv()  # Load OPENAI_API_KEY from .env file
    
    # --- Configuration ---
    INPUT_FILE = "simplified_tweets.jsonl"
    OUTPUT_FILE = "output_data.jsonl"
    ERROR_FILE = "error_items.jsonl"
    BATCH_SIZE = 5  # Small batch size for safety
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('classification.log'),
            logging.StreamHandler()
        ]
    )
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # --- Model Initialization ---
    config = APIModelConfig(model_name="gpt4o", temperature=0.0, max_tokens=4096)
    model = LiteLLMModel(config=config, name="gpt4o-classifier", logger=logging.getLogger(__name__))
    
    # --- Data Loading and Processing ---
    data = load_data(INPUT_FILE)
    if not data:
        logging.error("No valid data found in input file")
        return
    
    system_prompt = get_classification_prompt()
    logging.info(f"Loaded {len(data)} records. Starting classification in batches of {BATCH_SIZE}.")
    
    successful_count = 0
    error_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out, \
         open(ERROR_FILE, 'w', encoding='utf-8') as f_error:
        
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Processing Batches"):
            batch = data[i:i+BATCH_SIZE]
            user_prompt_content = format_batch_for_prompt(batch)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content},
            ]
            
            try:
                response = model.query(messages=messages)
                content = response.choices[0].message.content
                
                # Parse and save results
                results = parse_model_response(content, batch)
                
                for result in results:
                    if result.get("hate_speech") == -1:  # Error in parsing
                        # Add original text to error file for debugging
                        original_item = next((item for item in batch if item["ID"] == result["ID"]), None)
                        if original_item:
                            error_item = {**result, "original_text": original_item.get("Text")}
                            f_error.write(json.dumps(error_item) + "\n")
                        error_count += 1
                    else:
                        f_out.write(json.dumps(result) + "\n")
                        successful_count += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)  # Increased from 0.5 to 1 second
                
            except Exception as e:
                logging.error(f"Failed to process batch starting at index {i}: {e}")
                # Save failed batch items to error file
                for item in batch:
                    error_item = {
                        "ID": item.get("ID"),
                        "Text": item.get("Text"),
                        "error": str(e)
                    }
                    f_error.write(json.dumps(error_item) + "\n")
                error_count += len(batch)
    
    logging.info(f"Classification complete.")
    logging.info(f"Successfully processed: {successful_count} items")
    logging.info(f"Errors: {error_count} items")
    logging.info(f"Results saved to {OUTPUT_FILE}")
    if error_count > 0:
        logging.info(f"Error items saved to {ERROR_FILE}")
    logging.info(f"Final Stats: {model.stats.model_dump_json(indent=2)}")

if __name__ == "__main__":
    main()

