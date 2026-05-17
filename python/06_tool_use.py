"""
Example 06: Tool use / function calling
Define Python functions across five categories and pass them as tools.
The model calls them automatically when a question requires computation
or a data lookup. Each category runs in its own focused conversation.
"""

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)


def _clean(value: str) -> str:
    """Strip Gemma-4 quote tokens that wrap string arguments."""
    return value.replace('<|"|>', "").strip()


# Arithmetic


def add_numbers(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


# Unit conversion


def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts a temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in degrees Celsius.
    """
    return celsius * 9 / 5 + 32


# Weather


def get_current_weather(city: str) -> str:
    """Returns a mock weather report for a given city.

    Args:
        city: The name of the city.
    """
    return f"The weather in {city} is sunny and 22°C."


# Country info


def get_country_capital(country: str) -> str:
    """Returns the capital city of a country.

    Args:
        country: The name of the country.
    """
    capitals = {
        "France": "Paris",
        "Germany": "Berlin",
        "Japan": "Tokyo",
        "Brazil": "Brasília",
        "Turkey": "Ankara",
    }
    country = _clean(country)
    return capitals.get(country, f"Capital of {country} not found.")


# BMI


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculates Body Mass Index (BMI).

    Args:
        weight_kg: Body weight in kilograms.
        height_m: Height in metres.
    """
    return round(weight_kg / (height_m**2), 2)


# Run each category in its own conversation

GROUPS = [
    (
        "Arithmetic",
        [add_numbers, multiply_numbers],
        ["What is 123 + 456?", "What is 7 multiplied by 8?"],
    ),
    (
        "Unit conversion",
        [celsius_to_fahrenheit],
        ["Convert 100 degrees Celsius to Fahrenheit."],
    ),
    (
        "Weather",
        [get_current_weather],
        ["What is the weather in Istanbul?"],
    ),
    (
        "Country info",
        [get_country_capital],
        ["What is the capital of Japan?", "What is the capital of Turkey?"],
    ),
    (
        "BMI",
        [calculate_bmi],
        ["What is the BMI for someone who is 1.75 m tall and weighs 70 kg?"],
    ),
]

with litert_lm.Engine(MODEL_PATH) as engine:
    for group_name, tools, queries in GROUPS:
        print(f"=== {group_name} ===")
        with engine.create_conversation(tools=tools) as conversation:
            for query in queries:
                print(f"Q: {query}")
                response = conversation.send_message(query)
                print(f"A: {response['content'][0]['text']}\n")
