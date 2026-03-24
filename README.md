# CSAI 422 – Assignment 3: Conversational Weather Agent


## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file

Create a file named `.env` in the project folder and add:

```env
API_KEY=your_api_key
BASE_URL=your_base_url
LLM_MODEL=your_model_name
WEATHER_API_KEY=your_weatherapi_key
```

### 3. Run the program

```bash
python conversational_agent.py
```

---

## Implementation Overview

This project implements a conversational agent capable of answering weather-related questions using external tools and reasoning strategies.

### 1. Basic Agent

The Basic agent supports simple weather queries using tool calling. It retrieves current weather or forecasts and returns a natural language response.

### 2. Chain of Thought Agent

The Chain of Thought agent extends the Basic agent by:

* Adding a calculator tool
* Using step-by-step reasoning
* Handling comparisons and mathematical operations

### 3. Advanced Agent

The Advanced agent includes:

* Safe tool execution with error handling
* Parallel and sequential tool execution
* Multi-step tool orchestration
* Structured JSON outputs for complex queries

### Bonus Evaluation

A comparison system is implemented to evaluate:

* Response quality
* Execution time
* Sequential vs parallel performance

---

## Example Conversations

### Basic Agent

**User:** What is the weather in Cairo?
**Assistant:** The current temperature in Cairo is 25°C with clear skies.

---

### Chain of Thought Agent

**User:** What is the temperature difference between Cairo and London?
**Assistant:**

* Cairo: 25°C
* London: 15°C
  Difference = 10°C
  So, Cairo is 10°C warmer than London.

---

### Advanced Agent

**User:** Compare Cairo, Dubai, and Paris and tell me which is warmer.

**Assistant (structured JSON):**

```json
{
  "query_type": "comparison",
  "locations": ["Cairo", "Dubai", "Paris"],
  "summary": "Dubai has the highest temperature among the three cities.",
  "tool_calls_used": ["get_current_weather"],
  "final_answer": "Dubai is the warmest city currently."
}
```

---

## Analysis

The three agents demonstrate increasing levels of reasoning and performance:

* The Basic agent is efficient for simple queries but limited in handling complex questions.
* The Chain of Thought agent improves accuracy by breaking down problems and performing calculations.
* The Advanced agent provides the best performance by:

  * Executing multiple tool calls in parallel
  * Handling multi-step reasoning
  * Producing structured outputs

Parallel execution significantly reduces response time when querying multiple locations simultaneously, compared to sequential execution.

---

## Challenges and Solutions

### 1. Handling API Errors

Some tool calls failed due to invalid inputs or network issues.
Solution: Implemented safe tool execution with error handling.

### 2. Managing Multi-step Reasoning

The agent initially failed to complete multi-step tasks.
Solution: Added iterative tool execution with a maximum iteration loop.

### 3. Structured Output Validation

Ensuring correct JSON format was challenging.
Solution: Implemented validation checks for required keys.

### 4. Environment Configuration

Setting up API keys caused issues initially.
Solution: Used a `.env` file and `python-dotenv` to manage configuration.

---

## References

* OpenAI API Function Calling Documentation:
  https://platform.openai.com/docs/guides/function-calling

* OpenAI API Structured Outputs Documentation:
  https://platform.openai.com/docs/guides/structured-outputs

* WeatherAPI Documentation:
  https://www.weatherapi.com/docs/

* OpenRouter API Documentation:
  https://openrouter.ai/docs

---

## Notes

This project uses OpenRouter as the LLM provider, which is compatible with the OpenAI API. Therefore, OpenAI documentation was used as the primary reference for tool calling and structured outputs.

The `.env` file is not included in this repository for security reasons.
