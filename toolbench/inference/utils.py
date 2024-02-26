import gc
import abc
import numpy as np
import math
from typing import Iterable
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# For DFS
def softmax_bias(answers,temperature=1):

    sums = 0.0
    answers = [ 10**((cont/temperature)/400) for cont in answers]
    for cont in answers:
        assert type(cont) == float or type(cont) == int
        sums += cont
    answers = [ cont/sums for cont in answers]
    return np.array(answers)

def compute_epsilon_new_node(p_new_node):
    '''
    根据公式换算delta
    '''
    delta = 400 * math.log10(p_new_node /(1-p_new_node))
    return 1000 + delta

# For prediction parsing, into ReACT format
def react_parser(string):
    thought = [string[string.find("Thought: ") + len("Thought: "): string.find("\nAction: ")]]
    action = [string[string.find("Action: ") + len("Action: "): string.find("\nAction Input: ")]]
    action_input = [string[string.find("Action Input: ") + len("Action Input: "):]]
    return thought[0], action[0], action_input[0]

# For IO presentation
class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""
    
    @abc.abstractmethod
    def return_output(self, output_stream):
        """Return output."""

class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)
    
    def return_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                pre = now
        return " ".join(output_text)
