import json
import re
import torch
import transformers
import transformers.models.llama.modeling_llama
from functools import partial


def process_system_message(system_message, functions):
    assert "with a function call to actually excute your step." in system_message
    # we find that following ReACT format and merging the thought node and function call node is easier for model to learn to integrate the action input json string in its prediction than learn to predict a json string directly.
    system_message = system_message.replace("with a function call to actually excute your step.", "with a function call to actually excute your step. Your output should follow this format:\nThought:\nAction\nAction Input:\n")
    # add all the function dicts in the prompt.
    system_message = system_message + "\nSpecifically, you have access to the following APIs: " + str(functions)
    return system_message

def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name
    
def process_retrieval_ducoment(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = (doc.get('category_name', '') or '') + ', ' + \
        (doc.get('tool_name', '') or '') + ', ' + \
        (doc.get('api_name', '') or '') + ', ' + \
        (doc.get('api_description', '') or '') + \
        ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
        ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
        ', return_schema: ' + json.dumps(doc.get('template_response', ''))
        corpus2tool[(doc.get('category_name', '') or '') + ', ' + \
        (doc.get('tool_name', '') or '') + ', ' + \
        (doc.get('api_name', '') or '') + ', ' + \
        (doc.get('api_description', '') or '') + \
        ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
        ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
        ', return_schema: ' + json.dumps(doc.get('template_response', ''))] = doc['category_name'] + '\t' + doc['tool_name'] + '\t' + doc['api_name']
    return ir_corpus, corpus2tool
    
