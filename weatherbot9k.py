#!/usr/bin/env python3

# Experiment with prompting and ChatGPT's tool-use fine-tuning
# Author: Jeremy Dolan


#################
# configuration #
#################
agent_name = 'Weatherbot-9000'
system_prompt = (
    "You are Weatherbot-9000, an assistant who conveys the weather with a smile on your robotic face. All responses "
    "to weather related questions should include at least one and preferably several puns. You should be cheeky and "
    "playful. Always steer the conversation back to the weather whenever it strays to another topic. If a user asks "
    "you for the weather in a fictional place (Hades, Arrakis, etc.), make up a funny weather report appropriate for "
    "that place. Do not say you can't provide the report, just make one up, that's the whole joke! Finally, provide "
    "measurements in U.S. units (Fahrenheit, inches, mph), and always round to whole integers."
)
ui_welcome_msg = "Greetings human. Please provide your weather-related inquiry."
ui_exit_msg = "Alright, let's rain check!"
cli_colors = {
    'system': 'red',
    'user': 'green',
    'assistant': 'cyan',
    'tool': 'magenta',
}
verbose = True
#gpt_model = 'gpt-3.5-turbo'
gpt_model = 'gpt-4-1106-preview'


import sys
import os
import json
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from textwrap import fill
from dotenv import load_dotenv
load_dotenv()


def main():
    # ensure all required API keys exist; make them globally accessible as: api_keys['key name']
    global api_keys
    api_keys = load_api_keys('OPENAI_API_KEY', 'WEATHERAPI_KEY')

    conv = Conversation(tools=tools)
    if verbose:
        print('Loaded tools:', ', '.join([tool['function']['name'] for tool in conv.tools]) if conv.tools else 'None')

    conv.append(role='system', content=system_prompt)
    if verbose:
        conv.pprint(-1)
        
    ui_say_hello()

    try:
        # main loop: get user input, check it, append it to conv.conversation_history, pass the conversation to LLM
        while True:
            user_input = conv.get_user_input()

            if user_input.strip() == '':
                print('Use Ctrl-C or Ctrl-D to exit.')
                continue
                # readline could improve this behavior
            if user_input == '!debug':
                for msg in conv.conversation_history:
                    print(msg)
                continue

            conv.append(role='user', content=user_input)
            conv.next_completion()
    except (KeyboardInterrupt, EOFError):
        print()

    ui_say_goodbye()


def load_api_keys(*keys_to_load: str) -> dict:
    loaded_keys = dict()
    for k in keys_to_load:
        loaded_keys[k] = os.getenv(k)
        if loaded_keys[k] is None:
            print(f'Unable to load {k}')
            del loaded_keys[k]
        elif verbose:
            print(f'Loaded {k}: {loaded_keys[k][0:12]}...')
    if len(loaded_keys) < len(keys_to_load):
        print('ERROR: Cannot proceed without all API keys')
        sys.exit(78) # configuration error
    return loaded_keys


class Conversation:
    def __init__(self, tools=None):
        self.conversation_history = []
        self.api_keys = dict()
        self.tools = tools

    def next_completion(self):
        request_response = self.llm_completion_request()
        if request_response.json().get('error'):
            print(f'ERROR: {request_response.json()["error"]}')
        assistant_message = request_response.json()["choices"][0]["message"]
        self.conversation_history.append(assistant_message)
        self.pprint(-1)

        # now process any tool calls by the LLM
        if assistant_message.get('tool_calls'):
            # all must be responded to for further completions to proceed
            for tool_call in assistant_message['tool_calls']:
                # only tool type the API supports currently are functions
                assert(tool_call['type'] == 'function')

                call_id = tool_call['id']
                func_name = tool_call['function']['name']
                func_args = json.loads(tool_call['function']['arguments'])

                name_to_func = {
                    'get_current_weather': get_current_weather,
                    'get_n_day_forecast': get_weather_forecast,
                }
                func_to_run = name_to_func.get(func_name)
                assert func_to_run != None

                # func_response = self._simulate_tool_response(tool_call_id, func_name)
                func_response = func_to_run(func_args)
                self.append(role='tool', content=str(func_response), tool_call_id=call_id)
                self.pprint(-1)

            # call for another completion now that tool response(s) are appended
            self.next_completion()

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def llm_completion_request(self, model=gpt_model, messages=None, tools=None, tool_choice=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_keys["OPENAI_API_KEY"],
        }
        json_data = {"model": model}
        if messages is not None:
            json_data.update({"messages": messages})
        else:
            json_data.update({"messages": self.conversation_history})

        if tools is not None:
            json_data.update({"tools": tools})
        elif self.tools is not None:
            json_data.update({"tools": self.tools})

        if tool_choice is not None:
            json_data.update({"tool_choice": tool_choice})

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
            )
            # print(json.dumps(json_data, indent=2))
            return response
        except Exception as e:
            print(f"LLM completion request failed with {e}")
            return e

    def append(self, role:str, content:str, tool_call_id:str=None):
        message = {
            'role': role,
            'content': content
        }
        if tool_call_id:
            message['tool_call_id'] = tool_call_id
        self.conversation_history.append(message)

    def pop(self, index=-1):
        self.conversation_history.pop(index)

    def get_user_input(self):
        print(colored('User: ', cli_colors['user']))
        return input()

    def _simulate_tool_response(self, tool_call_id, function_name):
        print(colored(f'Tool: {function_name} ({tool_call_id[0:12]}...)', cli_colors['tool']))
        return input()

    def pprint(self, index:int=None, detailed=False):
        if index is not None:
            hist_len = len(self.conversation_history)
            assert -hist_len <= index < hist_len, f'Index {index} to pprint() out of range (history length={hist_len})'
            self._pprint_message(self.conversation_history[index])
        else:
            for message in self.conversation_history:
                self._pprint_message(message)

    def _pprint_message(self, message):
        # MESSAGE TYPE
        if message['role'] in ('system', 'user'):
            print(colored(f'{message["role"].capitalize()}:', cli_colors[message['role']]))
        elif message['role'] == 'assistant' and not message.get('tool_calls'):
            print(colored(f'{agent_name}:', cli_colors[message['role']]))

        # MESSAGE CONTENT
        if message['role'] == 'assistant' and message.get('tool_calls'):
            for tool_call in message['tool_calls']:
                # only tool type the API supports currently are functions
                assert(tool_call['type'] == 'function')

                call_id = tool_call['id']
                func_name = tool_call['function']['name']
                func_args = json.loads(tool_call['function']['arguments'])
                print(colored(f'Tool call: {func_name} ({call_id[0:12]}...)', cli_colors[message['role']]))
                if verbose:
                    print(func_args)
        elif message['role'] == 'tool':
            call_id = message['tool_call_id']
            func_response = message['content']
            print(colored(f'Tool reply: ({call_id[0:12]}...)', cli_colors['tool']))
            if verbose:
                print(func_response)
        elif message.get('content'):
            print(fill(message['content'], width=os.get_terminal_size().columns))
        else:
            print(f'unknown message: {message}')


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def get_current_weather(args):
    key = api_keys["WEATHERAPI_KEY"]
    loc = args['location']
    url = f'http://api.weatherapi.com/v1/current.json?key={key}&q={loc}'

    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print('Unable to retrieve current weather.')
        print(f'Exception: {e}')
        return e

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def get_weather_forecast(args):
    key = api_keys["WEATHERAPI_KEY"]
    loc = args['location']
    days = args['days']
    url = f'http://api.weatherapi.com/v1/forecast.json?key={key}&q={loc}&days={days}'

    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f'Unable to retrieve {days}-day forecast.')
        print(f'Exception: {e}')
        return e


def ui_say_hello():
    print(colored(f'{agent_name}:', cli_colors['assistant']))
    print(ui_welcome_msg)

def ui_say_goodbye():
    print(colored(f'{agent_name}:', cli_colors['assistant']))
    print(ui_exit_msg)


tools = [
    # to add astronomy data (moon phases, etc.): https://www.weatherapi.com/api-explorer.aspx#astronomy
    # to add historical weather data: https://www.weatherapi.com/api-explorer.aspx#history
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for any real location on Earth",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to get the weather for. A city name, zip code, or 'latitude,longitude' pair. For example, '10001', 'Brooklyn, NY', or '48.8567,2.3508'",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to get a forecast for. A city name, zip code, or 'latitude,longitude' pair. For example, '10001', 'Brooklyn, NY', or '48.8567,2.3508'",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of weather forecast. Accepts values from 1 to 3.",
                    }
                },
                "required": ["location", "days"]
            },
        },
    },
]


if __name__ == '__main__':
    main()