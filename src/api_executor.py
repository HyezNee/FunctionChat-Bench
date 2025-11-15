import socket
import sys
import json
import openai
import torch._dynamo
import ast, re
from contextlib import closing
import signal
import subprocess
from contextlib import closing
import requests
from openai import AsyncOpenAI, OpenAI
import time
import os

torch._dynamo.config.cache_size_limit = 128

from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException

import vertexai

from src.openai_utils import retry_on_limit
from src.gemini_utils import (
    convert_messages_gemini,
    convert_gemini_to_response,
    call_gemini_model
)

import qwen_agent

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .utils import convert_system_prompt_into_user_prompt, combine_consecutive_user_prompts


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

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
    
# Inference w/ huggingface + vLLM
class Qwen3ModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, model_path):
        super().__init__(model, None)
        self.model_path = model_path
        self.port = find_free_port()
        self.host = "127.0.0.1"
        self.base_url = f"http://{self.host}:{self.port}"
        self.max_tokens = 4096

        print(f"[vLLM] launching on {self.base_url} with model='{self.model}' ...")
        cmd = [
            "vllm", "serve",
            self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.90",
            #"--max-model-len", "8192",
            "--trust-remote-code",
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes"
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        print("[vLLM] waiting for readiness ...")
        if not self.wait_until_ready(self.base_url, timeout_s=2000):
            raise RuntimeError("vLLM server did not become ready in time.")
        print("[vLLM] server is ready. issuing OpenAI-compatible request...")

        self.client = openai.OpenAI(base_url=f"{self.base_url}/v1", api_key="EMPTY")
        self.openai_chat_completion = self.client.chat.completions.create
    
    def wait_until_ready(self, base_url: str, timeout_s: int = 120):
        # vLLM API 서버 health 체크
        t0 = time.time()
        health = f"{base_url}/health"
        # health 없으면 /v1/models를 폴링해도 됨
        while time.time() - t0 < timeout_s:
            try:
                r = requests.get(health, timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def graceful_kill(self, grace_s: int = 10):
        if self.proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                self.proc.terminate()
            else:
                os.killpg(self.proc.pid, signal.SIGTERM)
            waited = 0
            while self.proc.poll() is None and waited < grace_s:
                time.sleep(0.5)
                waited += 0.5
            if self.proc.poll() is None:
                if os.name == "nt":
                    self.proc.kill()
                else:
                    os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception:
            pass

    def __del__(self):
        self.graceful_kill()
        print("[vLLM] done.")
    
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
                    tools=api_request['tools'],
                    tool_choice="auto",
                    max_completion_tokens=self.max_tokens,
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

class HyperCLOVAXModelAPI(AbstractModelAPIExecutor):
    def __init__(self, model, model_path):
        super().__init__(model, None)
        self.model_path = model_path
        self.port = find_free_port()
        self.host = "127.0.0.1"
        self.base_url = f"http://{self.host}:{self.port}"
        self.max_tokens = 4096

        print(f"[vLLM] launching on {self.base_url} with model='{self.model}' ...")
        cmd = [
            "vllm", "serve",
            self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.90",
            "--trust-remote-code",
            "--reasoning-parser", "hcx",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hcx",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        print("[vLLM] waiting for readiness ...")
        if not self.wait_until_ready(self.base_url, timeout_s=1000):
            raise RuntimeError("vLLM server did not become ready in time.")
        print("[vLLM] server is ready. issuing OpenAI-compatible request...")

        self.client = openai.OpenAI(base_url=f"{self.base_url}/v1", api_key="EMPTY")
        self.openai_chat_completion = self.client.chat.completions.create
    
    def wait_until_ready(self, base_url: str, timeout_s: int = 120):
        # vLLM API 서버 health 체크
        t0 = time.time()
        health = f"{base_url}/health"
        # health 없으면 /v1/models를 폴링해도 됨
        while time.time() - t0 < timeout_s:
            try:
                r = requests.get(health, timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def graceful_kill(self, grace_s: int = 10):
        if self.proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                self.proc.terminate()
            else:
                os.killpg(self.proc.pid, signal.SIGTERM)
            waited = 0
            while self.proc.poll() is None and waited < grace_s:
                time.sleep(0.5)
                waited += 0.5
            if self.proc.poll() is None:
                if os.name == "nt":
                    self.proc.kill()
                else:
                    os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception:
            pass

    def __del__(self):
        self.graceful_kill()
        print("[vLLM] done.")
    
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
                    tools=api_request['tools'],
                    tool_choice="auto",
                    max_completion_tokens=self.max_tokens,
                    stop=["<|im_end|><|endofturn|>", "<|im_end|><|stop|>"],
                    extra_body={
                        "skip_special_tokens": False,
                        # "force_reasoning": True
                    }
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
        if response_output["tool_calls"] == []:
            try:
                # <tool_call>\n{...json...}\n</tool_call> 파싱
                response_output['content'] = response_output['content'].split("<tool_call>")[-1].split("</tool_call")[0].strip()
                maybe_tools = json.loads(response_output['content'])
                response_output['tool_calls'] = [
                    {
                        'id': "hcx-functioncall-random-id",
                        'function': {
                            "arguments": maybe_tools.get('arguments', "{}"),
                            "name": maybe_tools['name'],
                            "type": "function",
                        }
                    }
                ]
            except Exception:
                pass
        return response_output

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
        # huggingface + vLLM
        elif model_name.lower().startswith('qwen3'):
            print("Load Qwen3 model with vLLM: ", model_name)
            return Qwen3ModelAPI(model_name, model_path)
        elif model_name.lower().startswith('hyperclovax'):
            print("Load HyperCLOVAX model with vLLM: ", model_name)
            return HyperCLOVAXModelAPI(model_name, model_path)