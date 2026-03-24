import os
import json
import time
import csv
import requests
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# Environment setup
# =========================================================
load_dotenv()

API_KEY = os.environ.get("API_KEY", os.getenv("OPTOGPT_API_KEY"))
BASE_URL = os.environ.get("BASE_URL", os.getenv("BASE_URL"))
LLM_MODEL = os.environ.get("LLM_MODEL", os.getenv("OPTOGPT_MODEL"))
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", os.getenv("WEATHER_API_KEY"))

if not API_KEY:
    raise ValueError("Missing API_KEY / OPTOGPT_API_KEY in environment.")
if not BASE_URL:
    raise ValueError("Missing BASE_URL in environment.")
if not LLM_MODEL:
    raise ValueError("Missing LLM_MODEL / OPTOGPT_MODEL in environment.")
if not WEATHER_API_KEY:
    raise ValueError("Missing WEATHER_API_KEY in environment.")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# =========================================================
# Part 1: Weather tools
# =========================================================
def get_current_weather(location):
    """Get the current weather for a location."""
    url = (
        f"http://api.weatherapi.com/v1/current.json"
        f"?key={WEATHER_API_KEY}&q={location}&aqi=no"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return json.dumps({"error": f"Request failed: {str(e)}"})

    if "error" in data:
        return json.dumps({"error": data["error"]["message"]})

    weather_info = data["current"]
    return json.dumps(
        {
            "location": data["location"]["name"],
            "region": data["location"].get("region", ""),
            "country": data["location"].get("country", ""),
            "temperature_c": weather_info["temp_c"],
            "temperature_f": weather_info["temp_f"],
            "condition": weather_info["condition"]["text"],
            "humidity": weather_info["humidity"],
            "wind_kph": weather_info["wind_kph"],
            "feelslike_c": weather_info.get("feelslike_c"),
            "feelslike_f": weather_info.get("feelslike_f"),
        }
    )


def get_weather_forecast(location, days=3):
    """Get a weather forecast for a location for a specified number of days."""
    if not isinstance(days, int):
        return json.dumps({"error": "days must be an integer"})
    if days < 1 or days > 10:
        return json.dumps({"error": "days must be between 1 and 10"})

    url = (
        f"http://api.weatherapi.com/v1/forecast.json"
        f"?key={WEATHER_API_KEY}&q={location}&days={days}&aqi=no&alerts=no"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return json.dumps({"error": f"Request failed: {str(e)}"})

    if "error" in data:
        return json.dumps({"error": data["error"]["message"]})

    forecast_days = data["forecast"]["forecastday"]
    forecast_data = []

    for day in forecast_days:
        forecast_data.append(
            {
                "date": day["date"],
                "max_temp_c": day["day"]["maxtemp_c"],
                "min_temp_c": day["day"]["mintemp_c"],
                "avg_temp_c": day["day"]["avgtemp_c"],
                "condition": day["day"]["condition"]["text"],
                "chance_of_rain": day["day"]["daily_chance_of_rain"],
            }
        )

    return json.dumps(
        {
            "location": data["location"]["name"],
            "region": data["location"].get("region", ""),
            "country": data["location"].get("country", ""),
            "forecast": forecast_data,
        }
    )

# =========================================================
# Part 2: Calculator tool
# =========================================================
def calculator(expression):
    """
    Evaluate a mathematical expression.
    Note: restricted eval for assignment use.
    """
    try:
        allowed_builtins = {}
        result = eval(expression, {"__builtins__": allowed_builtins}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# =========================================================
# Tool schemas for OpenAI API
# =========================================================
weather_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state, e.g., San Francisco, CA, "
                            "or a country, e.g., France"
                        ),
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Get the weather forecast for a location for a specific "
                "number of days"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state, e.g., San Francisco, CA, "
                            "or a country, e.g., France"
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "description": "The number of days to forecast (1-10)",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["location"],
            },
        },
    },
]

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate, "
                        "e.g., '2 + 2' or '5 * (3 + 2)'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}

cot_tools = weather_tools + [calculator_tool]
advanced_tools = cot_tools

available_functions = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast,
    "calculator": calculator,
}

# =========================================================
# System prompts
# =========================================================
basic_system_message = "You are a helpful weather assistant."

cot_system_message = """You are a helpful assistant that can answer questions
about weather and perform calculations.

When responding to complex questions, please follow these steps:
1. Think step-by-step about what information you need.
2. Break down the problem into smaller parts.
3. Use the appropriate tools to gather information.
4. Explain your reasoning clearly.
5. Provide a clear final answer.

For example, if someone asks about temperature conversions or
comparisons between cities, first get the weather data, then use the
calculator if needed, showing your work.
"""

advanced_system_message = """You are a helpful weather assistant that can use
weather tools and a calculator to solve multi-step problems.

Guidelines:
1. If the user asks about several independent locations, use multiple weather tool calls in parallel when appropriate.
2. If a question requires several steps, continue using tools until the task is completed.
3. If a tool fails, explain the issue clearly and continue safely when possible.
4. For complex comparison or calculation queries, prepare a structured final response.
"""

# =========================================================
# Utility helpers
# =========================================================
def normalize_message_for_history(message):
    """
    Convert SDK message objects to plain dicts when needed.
    """
    if isinstance(message, dict):
        return message

    msg = {
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", None),
    }

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        msg["tool_calls"] = []
        for tc in tool_calls:
            msg["tool_calls"].append(
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )
    return msg


def print_assistant_message(message):
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
    else:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)

    if role == "assistant" and content:
        print(f"\nAssistant: {content}\n")


def should_request_structured_output(user_input):
    keywords = [
        "compare",
        "difference",
        "average",
        "warmer",
        "colder",
        "higher",
        "lower",
        "forecast",
        "next",
        "calculate",
    ]
    lowered = user_input.lower()
    return any(word in lowered for word in keywords)

# =========================================================
# Part 1: Basic process_messages
# =========================================================
def process_messages(client, messages, tools=None, available_functions=None):
    """
    Process messages and invoke tools as needed.
    """
    tools = tools or []
    available_functions = available_functions or {}

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
    )
    response_message = response.choices[0].message
    messages.append(normalize_message_for_history(response_message))

    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            function_response = function_to_call(**function_args)

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        # Ask model again after tool results so it can produce final answer
        follow_up = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
        )
        follow_up_message = follow_up.choices[0].message
        messages.append(normalize_message_for_history(follow_up_message))

    return messages

# =========================================================
# Part 3.1: Safe tool execution helper
# =========================================================
def execute_tool_safely(tool_call, available_functions):
    """
    Execute a tool call with validation and error handling.
    Returns JSON string.
    """
    function_name = tool_call.function.name

    if function_name not in available_functions:
        return json.dumps(
            {
                "success": False,
                "error": f"Unknown function: {function_name}",
            }
        )

    try:
        function_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid JSON arguments: {str(e)}",
            }
        )

    try:
        function_response = available_functions[function_name](**function_args)
        return json.dumps(
            {
                "success": True,
                "function_name": function_name,
                "result": function_response,
            }
        )
    except TypeError as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid arguments: {str(e)}",
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }
        )

# =========================================================
# Part 3.2: Sequential and parallel tool execution
# =========================================================
def execute_tools_sequential(tool_calls, available_functions):
    """
    Execute tool calls one after another.
    """
    results = []
    for tool_call in tool_calls:
        safe_result = execute_tool_safely(tool_call, available_functions)
        tool_message = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": safe_result,
        }
        results.append(tool_message)
    return results


def execute_tools_parallel(tool_calls, available_functions, max_workers=4):
    """Execute independent tool calls in parallel."""
    if not tool_calls:
        return []

    def run_single_tool(tool_call):
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": execute_tool_safely(tool_call, available_functions),
        }

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tool_calls))) as executor:
        return list(executor.map(run_single_tool, tool_calls))


def compare_parallel_vs_sequential(tool_calls, available_functions):
    """
    Measure timing difference between sequential and parallel execution.
    """
    start = time.perf_counter()
    sequential_results = execute_tools_sequential(tool_calls, available_functions)
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    parallel_results = execute_tools_parallel(tool_calls, available_functions)
    parallel_time = time.perf_counter() - start

    speedup = sequential_time / parallel_time if parallel_time > 0 else None

    return {
        "sequential_results": sequential_results,
        "parallel_results": parallel_results,
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
    }

# =========================================================
# Part 3.3: Advanced process + multi-step workflow
# =========================================================
def process_messages_advanced(client, messages, tools=None, available_functions=None):
    """Send messages to the model and execute any returned tools in parallel."""
    tools = tools or []
    available_functions = available_functions or {}

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
    )

    response_message = response.choices[0].message
    messages.append(normalize_message_for_history(response_message))

    if response_message.tool_calls:
        tool_results = execute_tools_parallel(
            response_message.tool_calls,
            available_functions,
        )
        messages.extend(tool_results)

    return messages, response_message

# =========================================================
# Part 3.4: Structured final outputs
# =========================================================
required_output_keys = [
    "query_type",
    "locations",
    "summary",
    "tool_calls_used",
    "final_answer",
]

structured_output_prompt = """For complex comparison or calculation queries,
return the final answer as a valid JSON object with exactly these keys:
- query_type
- locations
- summary
- tool_calls_used
- final_answer
Do not include markdown fences.
"""


def validate_structured_output(response_text):
    """Validate the final structured JSON response."""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output: {str(e)}")

    for key in required_output_keys:
        if key not in parsed:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(parsed["locations"], list):
        raise ValueError("'locations' must be a list")

    if not isinstance(parsed["tool_calls_used"], list):
        raise ValueError("'tool_calls_used' must be a list")

    return parsed


def get_structured_final_response(client, messages):
    """
    Request a structured final response in JSON mode and validate it.
    """
    structured_messages = messages + [
        {
            "role": "system",
            "content": structured_output_prompt,
        }
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=structured_messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return validate_structured_output(content)

# =========================================================
# Conversation runners
# =========================================================
def run_conversation(client, system_message=basic_system_message, tools=None):
    """
    Run a conversation with the user.
    """
    if tools is None:
        tools = weather_tools

    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]

    print("Weather Assistant: Hello! I can help you with weather information.")
    print("Ask me about the weather anywhere!")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nWeather Assistant: Goodbye! Have a great day!")
            break

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        messages = process_messages(
            client,
            messages,
            tools,
            available_functions,
        )

        last_message = messages[-1]
        if last_message["role"] == "assistant" and last_message.get("content"):
            print(f"\nWeather Assistant: {last_message['content']}\n")

    return messages


def run_conversation_advanced(
    client,
    system_message=advanced_system_message,
    max_iterations=5,
):
    """
    Run a conversation that supports multi-step tool workflows.
    """
    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]

    print("Advanced Weather Assistant: Hello! Ask me complex weather questions.")
    print("I can compare cities, perform calculations, and return structured outputs.")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAdvanced Weather Assistant: Goodbye! Have a great day!")
            break

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        final_answer_printed = False

        for _ in range(max_iterations):
            messages, response_message = process_messages_advanced(
                client,
                messages,
                advanced_tools,
                available_functions,
            )

            if not response_message.tool_calls:
                # Model is done using tools
                last_message = messages[-1]
                if isinstance(last_message, dict) and last_message.get("role") == "assistant":
                    if should_request_structured_output(user_input):
                        try:
                            structured = get_structured_final_response(client, messages)
                            print("\nAdvanced Weather Assistant (Structured JSON):")
                            print(json.dumps(structured, indent=2))
                            print()
                        except Exception as e:
                            print(f"\nAdvanced Weather Assistant: {last_message.get('content', '')}")
                            print(f"(Structured output validation failed: {e})\n")
                    else:
                        print(f"\nAdvanced Weather Assistant: {last_message.get('content', '')}\n")

                final_answer_printed = True
                break

        if not final_answer_printed:
            print(
                "\nAdvanced Weather Assistant: I stopped after reaching the"
                " maximum number of tool iterations.\n"
            )

    return messages

# =========================================================
# Bonus: Evaluation helpers
# =========================================================
def get_single_turn_response(system_message, tools, user_query, advanced=False):
    """
    Run one user query through an agent and capture final response text.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query},
    ]

    start = time.perf_counter()

    if not advanced:
        messages = process_messages(client, messages, tools, available_functions)
        total_time = time.perf_counter() - start
        return {
            "messages": messages,
            "response": messages[-1].get("content", ""),
            "time": total_time,
        }

    for _ in range(5):
        messages, response_message = process_messages_advanced(
            client,
            messages,
            tools,
            available_functions,
        )
        if not response_message.tool_calls:
            break

    total_time = time.perf_counter() - start

    final_text = ""
    if isinstance(messages[-1], dict):
        final_text = messages[-1].get("content", "")

    return {
        "messages": messages,
        "response": final_text,
        "time": total_time,
    }


def extract_first_tool_calls_for_query(user_query):
    """
    Get the model's initial tool calls for a query so we can benchmark
    sequential vs parallel execution.
    """
    messages = [
        {"role": "system", "content": advanced_system_message},
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=advanced_tools,
    )

    response_message = response.choices[0].message
    return getattr(response_message, "tool_calls", None) or []


def save_evaluation_to_csv(filename, rows):
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "agent_type",
                "response",
                "response_time",
                "rating",
                "sequential_time",
                "parallel_time",
                "speedup",
            ],
        )

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)


def run_bonus_evaluation():
    print("\n=== Comparative Evaluation System ===")
    user_query = input("Enter a single weather query to evaluate: ").strip()

    print("\nRunning Basic agent...")
    basic_result = get_single_turn_response(
        basic_system_message, weather_tools, user_query, advanced=False
    )

    print("Running Chain of Thought agent...")
    cot_result = get_single_turn_response(
        cot_system_message, cot_tools, user_query, advanced=False
    )

    print("Running Advanced agent...")
    advanced_result = get_single_turn_response(
        advanced_system_message, advanced_tools, user_query, advanced=True
    )

    tool_calls = extract_first_tool_calls_for_query(user_query)
    perf = {
        "sequential_time": None,
        "parallel_time": None,
        "speedup": None,
    }

    if len(tool_calls) > 1:
        perf = compare_parallel_vs_sequential(tool_calls, available_functions)

    print("\n=== Results ===")
    print("\n[Basic]")
    print("Time:", round(basic_result["time"], 4), "seconds")
    print("Response:", basic_result["response"])

    print("\n[Chain of Thought]")
    print("Time:", round(cot_result["time"], 4), "seconds")
    print("Response:", cot_result["response"])

    print("\n[Advanced]")
    print("Time:", round(advanced_result["time"], 4), "seconds")
    print("Response:", advanced_result["response"])

    if perf["sequential_time"] is not None:
        print("\n=== Sequential vs Parallel Tool Timing ===")
        print("Sequential time:", round(perf["sequential_time"], 4), "seconds")
        print("Parallel time:", round(perf["parallel_time"], 4), "seconds")
        print("Speedup:", round(perf["speedup"], 4), "x")
    else:
        print("\nNo multi-tool benchmark available for this query.")

    def get_rating(agent_name):
        while True:
            try:
                rating = int(input(f"Rate {agent_name} response (1-5): ").strip())
                if 1 <= rating <= 5:
                    return rating
                print("Please enter a number from 1 to 5.")
            except ValueError:
                print("Invalid input. Please enter a number from 1 to 5.")

    basic_rating = get_rating("Basic")
    cot_rating = get_rating("Chain of Thought")
    advanced_rating = get_rating("Advanced")

    rows = [
        {
            "query": user_query,
            "agent_type": "Basic",
            "response": basic_result["response"],
            "response_time": basic_result["time"],
            "rating": basic_rating,
            "sequential_time": perf["sequential_time"],
            "parallel_time": perf["parallel_time"],
            "speedup": perf["speedup"],
        },
        {
            "query": user_query,
            "agent_type": "Chain of Thought",
            "response": cot_result["response"],
            "response_time": cot_result["time"],
            "rating": cot_rating,
            "sequential_time": perf["sequential_time"],
            "parallel_time": perf["parallel_time"],
            "speedup": perf["speedup"],
        },
        {
            "query": user_query,
            "agent_type": "Advanced",
            "response": advanced_result["response"],
            "response_time": advanced_result["time"],
            "rating": advanced_rating,
            "sequential_time": perf["sequential_time"],
            "parallel_time": perf["parallel_time"],
            "speedup": perf["speedup"],
        },
    ]

    save_evaluation_to_csv("agent_evaluation_results.csv", rows)
    print("\nSaved results to agent_evaluation_results.csv")

# =========================================================
# Main menu
# =========================================================
if __name__ == "__main__":
    print("Choose an option:")
    print("1: Basic agent")
    print("2: Chain of Thought agent")
    print("3: Advanced agent")
    print("4: Bonus comparative evaluation")

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        run_conversation(client, basic_system_message, weather_tools)
    elif choice == "2":
        run_conversation(client, cot_system_message, cot_tools)
    elif choice == "3":
        run_conversation_advanced(client, advanced_system_message)
    elif choice == "4":
        run_bonus_evaluation()
    else:
        print("Invalid choice. Defaulting to Basic agent.")
        run_conversation(client, basic_system_message, weather_tools)