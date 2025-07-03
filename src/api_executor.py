import sys
import json
import openai

from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException

import vertexai

from src.openai_utils import retry_on_limit
from src.gemini_utils import (
    convert_messages_gemini,
    convert_tools_gemini,
    convert_gemini_to_response,
    call_gemini_model
)

import qwen_agent

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
from .utils import convert_system_prompt_into_user_prompt, combine_consecutive_user_prompts


class AbstractModelAPIExecutor:
    """
    A base class for model API executors that defines a common interface for making predictions.
    This class should be inherited by specific API executor implementations.

    Attributes:
        model (str): The model identifier.
        api_key (str): The API key for accessing the model.
    """
    def __init__(self, model, api_key):
        """
        Initializes the API executor with the specified model and API key.

        Parameters:
            model (str): The model identifier.
            api_key (str): The API key for accessing the model.
        """
        self.model = model
        self.api_key = api_key

    def predict(self):
        """
        A method to be implemented by subclasses that makes a model prediction.
        Raises a NotImplementedError if called on an instance of this abstract class.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class OpenaiModelAzureAPI(AbstractModelAPIExecutor):
    def __init__(self, model, api_key, api_base, api_version):
        """
        Initialize the OpenaiModelAzureAPI class.

        Parameters:
        model (str): The name of the model to use.
        api_key (str): The API key for authenticating with Azure OpenAI.
        api_base (str): The base URL for the Azure OpenAI API endpoint.
        api_version (str): The version of the Azure OpenAI API to use.
        """
        super().__init__(model, api_key)  # 수정된 부분
        self.client = openai.AzureOpenAI(azure_endpoint=api_base,
                                         api_key=api_key,
                                         api_version=api_version)
        self.openai_chat_completion = retry_on_limit(self.client.chat.completions.create)

    def predict(self, api_request):
        """
        A method get model predictions for a request.

        Parameters:
        api_request (dict): The API request data for making predictions.
        """
        response = None
        try_cnt = 0
        while True:
            try:
                response = self.openai_chat_completion(
                    model=self.model,
                    temperature=api_request['temperature'],
                    messages=api_request['messages']
                )
                response = response.model_dump()
            except Exception as e:
                print(f".. retry api call .. {try_cnt}")
                try_cnt += 1
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                continue
            else:
                break
        return response


class OpenaiModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, api_key, use_eval=False):
        """
        Initialize the OpenaiModelAPI class.

        Parameters:
        model (str): The name of the model to use.
        api_key (str): The API key for authenticating with OpenAI.
        use_eval (bool): Whether the API is for evaluation.
        """
        super().__init__(model, api_key)  # 수정된 부분
        self.client = openai.OpenAI(api_key=api_key)
        self.openai_chat_completion = self.client.chat.completions.create
        if use_eval is True:
            self.predict = self.predict_eval
        else:
            self.predict = self.predict_tool

    def predict_tool(self, api_request):
        """
        A method get model predictions for a request.

        Parameters:
        api_request (dict): The API request data for making predictions.
        """
        response = None
        try_cnt = 0
        while True:
            try:
                response = self.openai_chat_completion(
                    model=self.model,
                    temperature=api_request['temperature'],
                    messages=api_request['messages'],
                    tools=api_request['tools']
                )
                response = response.model_dump()
            except KeyError as e:
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                sys.exit(1)
            except Exception as e:
                print(f".. retry api call .. {try_cnt}")
                try_cnt += 1
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                continue
            else:
                break
        response_output = response['choices'][0]['message']
        return response_output

    def predict_eval(self, api_request):
        """
        A method get model predictions for a requests for evaluation purposes.

        Parameters:
        api_request (dict): The API request data for making predictions.
        """
        response = None
        try_cnt = 0
        while True:
            try:
                response = self.openai_chat_completion(
                    model=self.model,
                    temperature=api_request['temperature'],
                    messages=api_request['messages']
                )
                response = response.model_dump()
            except KeyError as e:
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                sys.exit(1)
            except Exception as e:
                print(f".. retry api call .. {try_cnt}")
                try_cnt += 1
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                continue
            else:
                break
        return response


class SolarModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, api_key, base_url):
        """
        Initialize the SolarModelAPI class.

        Parameters:
        model (str): The name of the model to use.
        api_key (str): The API key for authenticating with OpenAI.
        base_url (str): The base URL for the Solar API endpoint.
        """
        super().__init__(model, api_key)
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.openai_chat_completion = retry_on_limit(self.client.chat.completions.create)

    def predict(self, api_request):
        """
        A method get model predictions for a request.

        Parameters:
        api_request (dict): The API request data for making predictions.
        """
        response = None
        try_cnt = 0
        while True:
            try:
                response = self.openai_chat_completion(
                    model=self.model,
                    temperature=api_request['temperature'],
                    messages=api_request['messages'],
                    tools=api_request['tools']
                )
                response = response.model_dump()
            except Exception as e:
                print(f".. retry api call .. {try_cnt}")
                try_cnt += 1
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                continue
            else:
                break
        response_output = response['choices'][0]['message']
        return response_output


class MistralModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, api_key):
        """
        Initialize the MistralModelAPI class.

        Parameters:
        model (str): The name of the model to use.
        api_key (str): The API key for authenticating with OpenAI.
        """
        super().__init__(model, api_key)
        print(f"model: {model}")
        print(f"api_key: {api_key}")
        self.client = MistralClient(api_key=api_key)
        self.openai_chat_completion = retry_on_limit(self.client.chat)

    def remove_content_for_toolcalls(self, messages):
        new_messages = []
        for msg in messages:
            if msg['role'] == 'assistant' and msg.get('content', None) and msg.get('tool_calls', None):
                msg['content'] = ""
            new_messages.append(msg)
        return new_messages

    def predict(self, api_request):
        """
        A method get model predictions for a request.

        Parameters:
        api_request (dict): The API request data for making predictions.
        """
        response = None
        try_cnt = 0
        while True:
            try:
                print("max tokens * 32768")
                print("temperature *", api_request['temperature'])
                print("messages *", api_request['messages'])
                print("tools *", api_request['tools'])
                response = self.openai_chat_completion(
                    model=self.model,
                    temperature=api_request['temperature'],
                    max_tokens=32768,
                    messages=api_request['messages'],
                    tools=api_request['tools']
                )
                response = response.model_dump()
                print(">> response *", json.dumps(response, ensure_ascii=False))
            except MistralAPIException as e:
                msg = json.loads(str(e).split('Message:')[1]).get('message')
                if msg == 'Assistant message must have either content or tool_calls, but not both.':
                    api_request['messages'] = self.remove_content_for_toolcalls(api_request['messages'])
                    print(f"[error] {msg}")
                    print(json.dumps(api_request['messages'], ensure_ascii=False))
                print(f".. retry api call .. {try_cnt} {msg} {msg == 'Assistant message must have either content or tool_calls, but not both.'}")
                try_cnt += 1
            except Exception as e:
                print(f".. retry api call .. {try_cnt}")
                try_cnt += 1
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                continue
            else:
                break
        response_output = response['choices'][0]['message']
        return response_output


class InhouseModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, api_key, base_url, model_path):
        """
        Initialize the MistralModelAPI class.

        Parameters:
        model (str): The name of the model to use.
        api_key (str): The API key for authenticating with OpenAI.
        base_url (str): The base URL for the Inhouse API endpoint.
        model_path (str): This is the information that needs to be passed in the header when calling the model API.
        """
        super().__init__(model, api_key)
        print(f"base_url: {base_url}")
        print(f"api_key: {api_key}")
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.openai_chat_completion = retry_on_limit(self.client.chat.completions.create)
        self.model_path = model_path

    def predict(self, api_request):
        """
        A method get model predictions for a request.

        Parameters:
        api_request (dict): The API request data for making predictions.
        """
        response = None
        try_cnt = 0
        while True:
            try:
                response = self.openai_chat_completion(
                    model=self.model_path,
                    temperature=api_request['temperature'],
                    messages=api_request['messages'],
                    tools=api_request['tools']
                )
                response = response.model_dump()
            except Exception as e:
                print(f".. retry api call .. {try_cnt}")
                try_cnt += 1
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                continue
            else:
                break
        response_output = response['choices'][0]['message']
        return response_output


class Qwen2ModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, api_key, base_url, model_path):
        super().__init__(model, api_key)
        print(f"base_url: {base_url}")
        print(f"api_key: {api_key}")
        if model_path is not None:
            model = model_path
        self.client = qwen_agent.llm.get_chat_model({
            'model': model,
            'model_server': base_url,
            'api_key': api_key
        })

    def predict(self, api_request):
        """
        A method get model predictions for a request.

        Parameters:
        api_request (dict): The API request data for making predictions.
        """
        messages = api_request['messages']
        tools = [tool['function'] for tool in api_request['tools']]
        responses = []

        for idx, msg in enumerate(messages):
            print(msg)
            if msg['role'] == 'tool':
                messages[idx]['role'] = 'function'
            if msg['role'] == 'assistant' and 'tool_calls' in msg:
                messages[idx]['function_call'] = msg['tool_calls'][0]['function']
        for responses in self.client.chat(messages=messages, functions=tools, stream=True):
            continue
        response = responses[0]
        tools = None
        if 'function_call' in response:
            tools = [{'id': "qwen2-functioncall-random-id", 'function': response['function_call'], 'type': "function", 'index': None}]
        return {
            "content": response['content'],
            "role": response['role'],
            "function_call": None,
            "tool_calls": tools,
            "tool_call_id": None,
            "name": None
        }


class GeminiModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, gcloud_project_id, gcloud_location):
        """
        Initialize the GeminiModelAPI class.

        Parameters:
        model (str): The name of the model to use.
        gcloud_project_id (str): The Google Cloud project ID, required for models hosted on Google Cloud.
        gcloud_location (str): The location of the Google Cloud project, required for models hosted on Google Cloud.
        """
        super().__init__(model, None)
        vertexai.init(project=gcloud_project_id, location=gcloud_location)

    def predict(self, api_request):
        """
        A method get model predictions for a request.

        Parameters:
        api_request (dict): The API request data for making predictions.
        """
        try_cnt = 0
        response = None

        gemini_temperature = api_request['temperature']
        gemini_system_instruction, gemini_messages = convert_messages_gemini(api_request['messages'])
        gemini_tools = convert_tools_gemini(api_request['tools'])

        while True:
            try:
                response = call_gemini_model(
                    gemini_model=self.model,
                    gemini_temperature=gemini_temperature,
                    gemini_system_instruction=gemini_system_instruction,
                    gemini_tools=gemini_tools,
                    gemini_messages=gemini_messages)
                gemini_response = response['candidates'][0]
                if "content" not in gemini_response and gemini_response["finish_reason"] == "SAFETY":
                    response_output = {"role": "assistant", "content": None, "tool_calls": None}
                else:
                    response_output = convert_gemini_to_response(gemini_response["content"])
            except Exception as e:
                print(f".. retry api call .. {try_cnt}")
                try_cnt += 1
                print(e)
                print(json.dumps(api_request['messages'], ensure_ascii=False))
                continue
            else:
                break
        return response_output



class LocalModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, model_path):
        super().__init__(model, None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.model_name = model

    def predict(self, api_request):
        messages = api_request['messages']
        tools = [tool['function'] for tool in api_request['tools']]
        if 'gemma-3' in self.model_name.lower():
            tool_prompt = '''Here is a list of functions in JSON format that you can invoke.
{functions}
When using tools, make calls in a JSON format:
{{"name": "tool_call_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}, ... (additional parallel tool calls as needed)'''
            messages[0]['content'] += tool_prompt.format(functions=tools)
            messages = convert_system_prompt_into_user_prompt(messages)
            messages = combine_consecutive_user_prompts(messages)
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        else:
            inputs = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        
        input_ids_len = inputs["input_ids"].shape[-1] # Get the length of the input tokens
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        generated_tokens = outputs[:, input_ids_len:] # Slice the output to get only the newly generated tokens
        decoded = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # {"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}
        tool_calls = None
        tool_content = None

        if '<tool_call>' in decoded:    # Qwen3, etc.
            tool_content = decoded.split('<tool_call>')[-1].replace('</tool_call>', '').strip()
        elif 'xlam' in self.model_name.lower() and '"name":' in decoded and 'arguments":' in decoded:
            if decoded.startswith('[{') and decoded.endswith('}]'):
                tool_content = decoded[1:-1]
            elif decoded.startswith('```json') and decoded.endswith('```'):
                tool_content = decoded[8:-4].strip()
        elif 'gemma-3' in self.model_name.lower() and '{"name":' in decoded and 'arguments":' in decoded:
            if decoded.startswith('```json') and decoded.endswith('```'):
                tool_content = decoded[8:-4].strip()

        if tool_content is not None:
            try:
                # json.loads로 감싸지 않은 경우도 있을 수 있음
                tool_json = json.loads(tool_content)
                # arguments 부분 str로 바꾸기
                tool_json['arguments'] = json.dumps(tool_json['arguments'], ensure_ascii=False)
            except Exception:
                tool_json = tool_content  # fallback

            tool_calls = [{
                'id': f"{self.model_name}-functioncall-random-id",
                'function': tool_json, 
                'type': "function",
                'index': None
            }]
        if tool_calls is None and 'qwen3' in self.model_name.lower():   # Qwen3-style think 태그 제거
            decoded = decoded.split("</think>")[-1].strip()

        return {
            "role": "assistant",
            "content": decoded if not tool_calls else "",  # tool 호출 시 content는 비워두기
            "tool_calls": tool_calls,
            "function_call": None,
            "name": None
        }
    

class VLLMModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, model_path):
        super().__init__(model, None)
        self.model_path = model_path
        self.llm = LLM(model=self.model_path,
                       dtype=torch.bfloat16,
                       tensor_parallel_size=2) # gpu 개수
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model_name = model

    def predict(self, api_request):
        messages = api_request['messages']
        tools = [tool['function'] for tool in api_request['tools']]
        if 'gemma-3' in self.model_name.lower():
            tool_prompt = '''Here is a list of functions in JSON format that you can invoke.
{functions}
When using tools, make calls in a JSON format:
{{"name": "tool_call_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}, ... (additional parallel tool calls as needed)'''
            messages[0]['content'] += tool_prompt.format(functions=tools)
            messages = convert_system_prompt_into_user_prompt(messages)
            messages = combine_consecutive_user_prompts(messages)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False
            )
        
        # Generate with vLLM
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
        outputs = self.llm.generate(prompt, sampling_params, use_tqdm=False)
        decoded = outputs[0].outputs[0].text.strip()
        
        # {"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}
        tool_calls = None
        tool_content = None

        if '<tool_call>' in decoded:    # Qwen3, etc.
            tool_content = decoded.split('<tool_call>')[-1].replace('</tool_call>', '').strip()
        elif 'xlam' in self.model_name.lower() and '"name":' in decoded and 'arguments":' in decoded:
            if decoded.startswith('[{') and decoded.endswith('}]'):
                tool_content = decoded[1:-1]
            elif decoded.startswith('```json') and decoded.endswith('```'):
                tool_content = decoded[8:-4].strip()
            else:
                tool_content = decoded
        elif 'gemma-3' in self.model_name.lower() and '"name":' in decoded and 'arguments":' in decoded:
            if decoded.startswith('```json') and decoded.endswith('```'):
                tool_content = decoded[8:-4].strip()
            else:
                tool_content = decoded

        if tool_content is not None:
            try:
                # json.loads로 감싸지 않은 경우도 있을 수 있음
                tool_json = json.loads(tool_content)
                # arguments 부분 str로 바꾸기
                tool_json['arguments'] = json.dumps(tool_json['arguments'], ensure_ascii=False)
            except Exception:
                tool_json = tool_content  # fallback

            tool_calls = [{
                'id': f"{self.model_name}-functioncall-random-id",
                'function': tool_json, 
                'type': "function",
                'index': None
            }]
        if tool_calls is None and 'qwen3' in self.model_name.lower():   # Qwen3-style think 태그 제거
            decoded = decoded.split("</think>")[-1].strip()

        return {
            "role": "assistant",
            "content": decoded if not tool_calls else "",  # tool 호출 시 content는 비워두기
            "tool_calls": tool_calls,
            "function_call": None,
            "name": None
        }


class APIExecutorFactory:
    """
    A factory class to create model API executor instances based on the model name.
    """

    @staticmethod
    def get_model_api(model_name, api_key=None, model_path=None, base_url=None, gcloud_project_id=None, gcloud_location=None):
        """
        Creates and returns an API executor for a given model by identifying the type of model and initializing the appropriate API class.

        Parameters:
            model_name (str): The name of the model to be used. It determines which API class is instantiated.
            api_key (str, optional): The API key required for authentication with the model's API service.
            model_path (str, optional): The path to the model, used for in-house models.
            base_url (str, optional): The base URL of the API service for the model.
            gcloud_project_id (str, optional): The Google Cloud project ID, required for models hosted on Google Cloud.
            gcloud_location (str, optional): The location of the Google Cloud project, required for models hosted on Google Cloud.

        Returns:
            An instance of an API executor for the specified model.

        Raises:
            ValueError: If the model name is not supported.

        The method uses the model name to determine which API executor class to instantiate and returns an object of that class.
        """
        if model_name == 'inhouse':  # In-house developed model
            return InhouseModelAPI(model_name, api_key, base_url=base_url, model_path=model_path)
        elif model_name.lower().startswith('qwen2'):  # Upstage developed model
            return Qwen2ModelAPI(model_name, api_key=api_key, base_url=base_url, model_path=model_path)
        elif model_name.lower().startswith('solar'):  # Upstage developed model
            return SolarModelAPI(model_name, api_key=api_key, base_url=base_url)
        elif model_name.lower().startswith('gpt'):  # OpenAI developed model
            return OpenaiModelAPI(model_name, api_key)
        elif model_name.startswith('mistral'):  # Mistral developed model
            return MistralModelAPI(model_name, api_key)
        elif model_name.startswith('gemini'):  # Google developed model
            return GeminiModelAPI(model_name, gcloud_project_id=gcloud_project_id, gcloud_location=gcloud_location)
        elif model_name.lower().startswith('vllm'): # local model: vLLM
            print("Load local model with vLLM: ", model_name)
            return VLLMModelAPI(model_name, model_path)
        else:
            print("Load local model: ", model_name)
            return LocalModelAPI(model_name, model_path)
