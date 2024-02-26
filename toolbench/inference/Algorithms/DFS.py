import json
import random
import re
import torch

from Algorithms.base_search import base_search_method
from collections import Counter
from copy import deepcopy
from LLM_rank.rank_candidate import sum_based_rankn, rank2_subfix
from Prompts.Judgment_prompts import JUDGMENT_SYSTEM_PROMPT, JUDGMENT_USER_PROMPT
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
from Prompts.Tree_search_prompts import DIVERSITY_PROMPT
from Tree.Tree import my_tree, tree_node


STATE_NORMAL = 'normal'
STATE_EXCEEDED = 'exceeded'
STATE_USELESS = 'useless'


def standardize(x):
    return ' '.join(x.strip().split())


def compare_cache_item(function_name, function_input, cache_item):
    cache_item_json = json.loads(cache_item)
    if cache_item_json['function_name'] != function_name:
        return False
    cache_item_input_json = cache_item_json['function_input']
    function_input_json = json.loads(function_input)
    cache_item_argument_names = set(cache_item_input_json.keys())
    function_argument_names = set(function_input_json.keys())
    if cache_item_argument_names != function_argument_names:
        return False
    for arg in function_argument_names:
        if function_input_json[arg] != cache_item_input_json[arg]:
            return False
    return True


CLS = 0
PAD = 1
SEP = 2


def get_judgment(input_sequence, judgment_model, tokenizer):
    max_length = 2048
    S0 = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(y)) for y in input_sequence]
    S = [S0[3], S0[0], S0[2], S0[1]]
    input_ids = []
    for j, s in enumerate(S):
        if j == 0:
            input_ids += [CLS] + s + [SEP]
        else:
            input_ids += [SEP] + s + [SEP]
    if len(input_ids) > max_length:
        input_ids = input_ids[ : max_length]
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        logits = judgment_model(
            input_ids=input_ids.to(judgment_model.device)
        )[0]
        scores = torch.exp(logits[ : , 1]) / (torch.exp(logits[ : , 1]) + torch.exp(logits[ : , 0]))
    score = list(scores.cpu().numpy())[0]
    if score > 0.5:
        judgment = 1
    else:
        judgment = 0
    print('Judgment:', input_sequence, judgment)
    return judgment

class DFS_tree_search(base_search_method):

    def __init__(self, llm, io_func, process_id=0, callbacks=None):
        super(DFS_tree_search, self).__init__(
            llm, io_func, process_id, callbacks)
        """Depth-first search. 
        with_filter=True: Every time a child node is generated, choose the best multiple iterations to go.
        with_filter=False: Do as Preorder traversal.
        """
        self.io_func = io_func
        self.llm = llm
        self.process_id = process_id
        self.restart()

        self.callbacks = callbacks if callbacks is not None else []

        self.th_blacklist = 1
        self.judgment_function = [{
            'name': 'give_judgment',
            'description': 'Judge whether the returned result contains any valuable information other than error messages.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reason': {
                        'type': 'string',
                        'description': 'explain your answer.'
                    },
                    'judgment': {
                        'type': 'string',
                        'enum': ['Yes', 'No']
                    }
                },
                'required': ['reason', 'judgment']
            }
        }]
        self.summary_function = [{
            'name': 'give_response',
            'description': 'Give a response to user about the related information extracted from the returned results of the external APIs.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'response': {
                        'type': 'string'
                    }
                },
                'required': ['response']
            }
        }]

    def restart(self):
        self.status = 0
        self.terminal_node = []
        self.give_up_node = []
        self.now_expand_num = 0
        self.query_count = 0
        self.total_tokens = 0

    def send_agent_chain_end(self, depth, agent_block_ids, chain_block_ids):
        for i in range(len(self.callbacks)):
            callback = self.callbacks[i]
            callback.on_chain_end(
                depth=depth,
                block_id=chain_block_ids[i]
            )
            if i < len(agent_block_ids):
                callback.on_agent_end(
                    depth=depth,
                    block_id=agent_block_ids[i]
                )

    def to_json(self, answer=False, process=True):

        if process:
            json_obj = {
                "win": self.status == 1,
                "tree": self.tree.to_json_recursive(),
                # "forward_args": self.forward_args,
                "compare_candidates": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:  # has answer
                    json_obj["compare_candidates"].append(
                        node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "final_answer": "",
                "finish_type": "give_answer",
                "function": self.old_functions,
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_answer"
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
            # do not have final answer, look for give_up
            if json_obj["answer_generation"]["valid_data"] == False:
                if len(self.give_up_node) > 0:
                    random_pos = random.randint(0, len(self.give_up_node) - 1)
                    choose_give_up_node = self.give_up_node[random_pos]
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_up"
                    json_obj["answer_generation"]["final_answer"] = choose_give_up_node.description
                    json_obj["answer_generation"]["train_messages"] = choose_give_up_node.get_train_messages_from_this_node()
        
        if self.summary_message is not None:
            new_cache = []
            for x in self.cache:
                y = deepcopy(self.cache[x])
                xx = json.loads(x)
                y.update(xx)
                new_cache.append(y)
            new_cache.sort(key=lambda x: x['timestamp'])
            new_messages = deepcopy(self.tree.root.messages)
            for y in new_cache:
                new_messages.append({
                    'role': 'assistant',
                    'content': None,
                    'function_call': {
                        'name': y['function_name'],
                        'arguments': json.dumps(y['function_input'], ensure_ascii=False)
                    }
                })
                new_messages.append({
                    'role': 'function',
                    'name': y['function_name'],
                    'content': y['observation']
                })
            final_answer = {'return_type': 'give_answer', 'final_answer': self.summary_message}
            final_answer = json.dumps(final_answer, ensure_ascii=False)
            new_messages.append({
                'role': 'assistant',
                'content': None,
                'function_call': {
                    'name': 'Finish',
                    'arguments': final_answer
                }
            })
            json_obj["answer_generation"]["valid_data"] = True
            json_obj["answer_generation"]["finish_type"] = "give_answer"
            json_obj["answer_generation"]["final_answer"] = final_answer
            if not ("train_messages" in json_obj["answer_generation"]):
                json_obj["answer_generation"]["train_messages"] = []
            json_obj["answer_generation"]["train_messages"].append(new_messages)
        return json_obj

    def summarize(self):
        summarize_system_prompt = 'Given a user query which should be resolved using external APIs. Then, given the documentation of the external APIs, the input parameters for calling the APIs and the returned results given by the APIs. Note that for each given API, there may be multiple calls of the API with different input parameters. Please generate a response to the user which includes the key information related to the user query from the returned results.'
        user_query = self.tree.root.messages[1]['content'].strip().split('\n')[0]
        summarize_user_prompt = 'User query: %s\n\n' % user_query
        new_cache = {}
        for x in self.cache:
            y = deepcopy(self.cache[x])
            xx = json.loads(x)
            y['function_input'] = xx['function_input']
            func = xx['function_name']
            if not func in new_cache:
                new_cache[func] = []
            new_cache[func].append(y)
        for func in new_cache:
            new_cache[func].sort(key=lambda x: x['timestamp'])
            api_documentation = None
            for f in self.old_functions:
                if f['name'] == func:
                    api_documentation = json.dumps(f, ensure_ascii=False)
                    break
            summarize_user_prompt += 'Below are the documentation, input parameters and returned results of the API "%s":\nDocumentation: %s\n' % (func, api_documentation)
            for i, y in enumerate(new_cache[func]):
                summarize_user_prompt += 'Input Parameters (%d): %s\nReturned Result (%d): %s\n' % (i + 1, json.dumps(y['function_input'], ensure_ascii=False), i + 1, y['observation'])
        summarize_user_prompt += '\nNow please generate a response to the user which includes the key information related to the user query from the returned results given above. Do not directly include the names of the APIs in the generated response. Do not include any information which does not appear in the returned results. For the related information extracted from the returned results, it is best to list them as detailed as possible in the response. However, if the information is unrelated to the user query, just ignore it and do not list it in the response. Moreover, if any parts of the user query cannot be resolved, it is best to explain why these parts cannot be resolved.\n'
        summarize_messages = [{'role': 'system', 'content': summarize_system_prompt}, {'role': 'user', 'content': summarize_user_prompt}]
        self.llm.change_messages(summarize_messages)
        try:
            assert(len(summarize_user_prompt) < 25000)
            new_message, error_code, total_tokens = self.llm.parse(self.summary_function, process_id=self.process_id)
            summary_arguments = json.loads(new_message['function_call']['arguments'])
            summary = summary_arguments['response']
            return summary
        except Exception as e:
            print('summary generation failed because of', e)
            return None

    def start(self, single_chain_max_step, tree_beam_size, max_query_count, answer=1, with_filter=True, budget=0, use_prompt=False, use_blacklist=False, use_llm_blacklist=False, use_cache=False, judgment_model=None, tokenizer=None):
        """ single_chain_max_step: The maximum depth of the tree
            tree_beam_size: How many children nodes for one node are generated per layer
            answer = n means the Algo exits when find n "give_answer" nodes
            max_query_count: the Algo exits when OpenAI-query exists this value
            with_filter: This is the difference between normal DFS(with_filter=True) and DFSDT(with_filter=False). 
        """
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")
        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)

        system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        system = system.replace("{task_description}",
                                self.io_func.task_description)
        self.tree.root.messages.append({"role": "system", "content": system})

        user = FORMAT_INSTRUCTIONS_USER_FUNCTION
        user = user.replace("{input_description}",
                            self.io_func.input_description)
        self.tree.root.messages.append({"role": "user", "content": user})

        self.count_success = Counter()
        self.count_failed = Counter()
        
        self.init_budget = budget
        self.budget = budget
        # self.budget = 10000
        self.use_prompt = use_prompt
        self.use_blacklist = use_blacklist
        self.use_llm_blacklist = use_llm_blacklist
        self.use_cache = use_cache
        
        self.judgment_model = judgment_model
        self.tokenizer = tokenizer

        self.n_forbidden = 0
        self.curr_timestamp = 0

        self.cache = {}
        self.old_functions = deepcopy(self.io_func.functions)

        self.state = {}
        for f in self.io_func.functions:
            fname = f['name']
            self.state[fname] = STATE_NORMAL

        self.base_message = self.tree.root.messages[0]['content']
        
        if self.budget < 10000:
            self.remaining_budget = self.budget
        else:
            self.remaining_budget = 20
        
        if self.use_prompt:
            self.cost_message_template = 'Note that you should not use the function whose cost exceeds the remaining budget. The remaining budget is %d. The cost of the functions are listed below:'
            self.cost_message = self.cost_message_template % self.remaining_budget
            extra_message = '\n' + self.cost_message + '\n'
            self.toolcost_message = ''
            for i, f in enumerate(self.io_func.functions):
                fname = f['name']
                fcost = self.io_func.cost[fname]
                if fname == 'Finish':
                    continue
                self.toolcost_message += '%d. Function name: %s, cost: %d\n' % (i + 1, fname, fcost)
            extra_message += self.toolcost_message
            self.tree.root.messages[0]['content'] += extra_message
        
        removal = []
        for f in self.io_func.functions:
            fname = f['name']
            fcost = self.io_func.cost[fname]
            if self.io_func.budget_limit[fname] <= 0:
                print('Function %s forbidden since its visit times exceeded' % fname)
                removal.append(fname)
            elif fcost > self.budget:
                print('Function %s forbidden since its cost %d exceeds the remaining budget' % (fname, fcost))
                removal.append(fname)
        if len(removal) > 0:
            for fname in removal:
                self.remove_function(fname, STATE_EXCEEDED, self.tree.root)
                fcost = self.io_func.cost[fname]
            print('Remaining functions:', [f['name'] for f in self.io_func.functions])

        return_code = self.DFS(self.tree.root, single_chain_max_step, tree_beam_size, max_query_count, answer, with_filter)
        self.summary_message = self.summarize()
        print(self.summary_message)
        return return_code
    
    def __update(self, node, new_message):
        cnt = 1
        node.messages[0]['content'] = new_message
        for child in node.children:
            cnt += self.__update(child, new_message)
        return cnt
    
    def update_message(self, node):
        new_functions = []
        new_message = self.base_message
        if self.use_prompt:
            extra_message = '\n' + self.cost_message + '\n' + self.toolcost_message
            new_message += extra_message
        useless = []
        exceeded = []
        for f in self.old_functions:
            fname = f['name']
            if self.state[fname] == STATE_NORMAL:
                new_functions.append(f)
            elif self.state[fname] == STATE_EXCEEDED:
                exceeded.append(fname)
            else:
                assert(self.state[fname] == STATE_USELESS)
                useless.append(fname)
        if len(exceeded) > 0:
            message_exceeded = '\nNow some subfunctions of the available tools are forbidden since using them will exceed the budget limitation. The forbidden subfunctions are listed as below:\n'
            for i, fname in enumerate(exceeded):
                message_exceeded += '%d. %s\n' % (i + 1, fname)
            new_message += message_exceeded
        if len(useless) > 0:
            message_useless = '\nNow some subfunctions of the available tools are forbidden since they cannot provide valuable information or usually return error messages. The forbidden subfunctions are listed as below:\n'
            for i, fname in enumerate(useless):
                message_useless += '%d. %s\n' % (i + 1, fname)
            new_message += message_useless
        if len(new_functions) <= 1:
            new_message += '\nNow all subfunctions of the available tools are forbidden, and you should call the "Finish" function to end the task.\n'
        self.io_func.functions = new_functions
        cnt = self.__update(self.tree.root, new_message)
        print('%d nodes updated' % cnt)
    
    def update_cost_message(self, node):
        new_cost_message = self.cost_message_template % self.remaining_budget
        new_message = self.tree.root.messages[0]['content'].replace(self.cost_message, new_cost_message)
        cnt = self.__update(self.tree.root, new_message)
        print('%d nodes updated' % cnt)
        self.cost_message = new_cost_message

    def remove_function(self, function_name, new_state, node):
        if function_name in self.state:
            self.state[function_name] = new_state
            self.update_message(node)
    
    def load_from_cache(self, function_name, function_input):
        try:
            for cache_item in self.cache:
                if compare_cache_item(function_name, function_input, cache_item) and self.cache[cache_item]['cache_valid']:
                    return self.cache[cache_item]['observation'], 0
            return None, None
        except Exception as e:
            print('cache loaded failed because of', e)
            return None, None

    def write_to_cache(self, function_name, function_input, observation, cache_valid):
        try:
            function_input_json = json.loads(function_input)
            cache_item = json.dumps({'function_name': function_name, 'function_input': function_input_json})
            self.cache[cache_item] = {'observation': observation, 'cache_valid': cache_valid, 'timestamp': self.curr_timestamp}
            self.curr_timestamp += 1
        except Exception as e:
            print('cache writing failed because of', e)
    
    def is_blacklisted(self, name):
        if self.count_success[name] > 0:
            return False
        if self.count_failed[name] >= self.th_blacklist:
            return True
        return False

    def is_hallucination(self, name):
        function_names = [f['name'] for f in self.old_functions]
        if name in function_names:
            return False
        return True

    def get_judgment_messages(self, old_messages, function_name, function_input, observation):
        input_description = old_messages[1]['content'].strip().split('\n')[0]
        api_documentation = None
        for f in self.io_func.functions:
            if f['name'] == function_name:
                api_documentation = json.dumps(f, ensure_ascii=False)
                break
        judgment_messages = [{'role': 'system', 'content': JUDGMENT_SYSTEM_PROMPT}]
        user = JUDGMENT_USER_PROMPT
        user = user.replace('{input_description}', standardize(input_description))
        user = user.replace('{api_documentation}', standardize(api_documentation))
        user = user.replace('{function_input}', standardize(function_input))
        user = user.replace('{observation}', standardize(observation))
        judgment_messages.append({'role': 'user', 'content': user})
        return judgment_messages

    def DFS(self, now_node, single_chain_max_step, tree_beam_size, max_query_count, answer, with_filter=True):
        """Returns the number of grids to go back. When a child node of a node generates a final answer or give up, it should go back a few more grids
        In a sense, the larger this value is, the more diverse it is, and it is GreedySearch@n when it is enlarged to infinity.
        """

        # this two value declares the rate to go back, Algo degrades to CoT when the value=Inf
        
        if len(self.io_func.functions) <= 1:
            return 100000
        
        final_answer_back_length = 2
        prune_back_length = 2

        now_node.expand_num = self.now_expand_num
        self.now_expand_num += 1
        if now_node.get_depth() >= single_chain_max_step or now_node.pruned or now_node.is_terminal:
            if now_node.is_terminal:  # final answer
                self.status = 1
                self.terminal_node.append(now_node)
                return final_answer_back_length
            else:
                now_node.pruned = True
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                    return prune_back_length
                else:
                    return 1

        next_tree_split_nodes = []
        for i in range(tree_beam_size):
            temp_now_node = now_node

            """If a node have children now, We will prompt the model to generate different nodes than all the existing nodes"""
            delete_former_diversity_message = False
            diversity_message = None
            if len(temp_now_node.children) > 0:

                former_candidates_des = ""
                js_list = []
                for k, child in enumerate(temp_now_node.children):
                    temp_node = child
                    while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                        temp_node = temp_node.children[0]
                    if temp_node.node_type == "Action Input":
                        obj_dict = {
                            "name": temp_node.father.description,
                            "arguments": temp_node.description,
                            "function_output": temp_node.observation,
                            "mento-carlo-action-value": temp_node.compute_weight(),
                        }
                        js_list.append(obj_dict)

                if len(js_list) > 0:
                    former_candidates_des = former_candidates_des + \
                        f"{json.dumps(js_list,indent=2)}\n"
                    if temp_now_node.observation != "":
                        former_candidates_des = former_candidates_des + \
                            f"again, your former observation: {temp_now_node.observation}\n"
                    diverse_prompt = DIVERSITY_PROMPT
                    diverse_prompt = diverse_prompt.replace(
                        "{previous_candidate}", former_candidates_des)
                    diversity_message = {
                        "role": "user", "content": diverse_prompt}
                    temp_now_node.messages.append(diversity_message)

                    delete_former_diversity_message = True
            # on_chain_start
            now_depth = temp_now_node.get_depth() // 3
            chain_block_ids = [callback.on_chain_start(
                depth=now_depth,
                inputs=temp_now_node.messages
            ) for callback in self.callbacks]
            agent_block_ids = []
            self.llm.change_messages(temp_now_node.messages)
            # on_llm_start
            [callback.on_llm_start(
                depth=now_depth,
                messages=temp_now_node.messages
            ) for callback in self.callbacks]
            new_message, error_code, total_tokens = self.llm.parse(
                self.io_func.functions, process_id=self.process_id)
            # on_llm_end
            [callback.on_llm_end(
                depth=now_depth,
                response=new_message
            ) for callback in self.callbacks]
            self.query_count += 1
            self.total_tokens += total_tokens
            if self.query_count >= max_query_count:  # a big return value will cause the Algo to exit
                return 100000

            # We need to exclude the diversity_message, because it will influence child nodes
            if delete_former_diversity_message:
                temp_now_node.messages[-1]["valid"] = False

            # parse nodes from OpenAI-message like CoT method
            assert new_message["role"] == "assistant"
            if "content" in new_message.keys() and new_message["content"] != None:
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = new_message["content"]
                child_io_state = deepcopy(temp_now_node.io_state)

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                temp_now_node = temp_node

                if error_code != 0:
                    temp_now_node.observation_code = error_code
                    temp_now_node.pruned = True

            if "function_call" in new_message.keys():
                # on_agent_action
                agent_block_ids = [callback.on_agent_action(
                    depth=now_depth,
                    action=new_message["function_call"]["name"],
                    action_input=new_message["function_call"]["arguments"]
                ) for callback in self.callbacks]
                function_name = new_message["function_call"]["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(temp_now_node.io_state)

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)

                temp_node.print(self.process_id)
                temp_now_node = temp_node

                function_input = new_message["function_call"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(temp_now_node.io_state)
                # on_tool_start
                [callback.on_tool_start(
                    depth=now_depth,
                    tool_name=temp_now_node.description,
                    tool_input=function_input
                ) for callback in self.callbacks]

                if self.is_blacklisted(function_name):
                    observation = {'error': 'Useless API', 'message': 'This API is forbidden since they cannot provide valuable information or usually return error messages.'}
                    if len(self.io_func.functions) <= 1:
                        observation['message'] += ' Since all subfunctions of the available tools are forbidden, you should call the "Finish" function to end the task.'
                    observation = json.dumps(observation)
                    status = 1
                    print('Function %s is useless. Success:' % function_name, self.count_success, 'Failed:', self.count_failed)
                    call_valid = False
                elif self.is_hallucination(function_name):
                    observation = {'error': 'API does not exist', 'message': 'This API does not exist.'}
                    if len(self.io_func.functions) <= 1:
                        observation['message'] += ' Since all subfunctions of the available tools are forbidden, you should call the "Finish" function to end the task.'
                    observation = json.dumps(observation)
                    status = 1
                    call_valid = False
                elif (self.io_func.cost[function_name] > self.budget) or (self.io_func.budget_limit[function_name] <= 0):
                    observation = {'error': 'Budget Limitation Exceeded', 'message': 'This API is forbidden since using it will exceed the budget limitation.'}
                    if len(self.io_func.functions) <= 1:
                        observation['message'] += ' Since all subfunctions of the available tools are forbidden, you should call the "Finish" function to end the task.'
                    observation = json.dumps(observation)
                    status = 1
                    call_valid = False
                else:
                    if self.use_cache:
                        observation, status = self.load_from_cache(function_name, function_input)
                    else:
                        observation, status = None, None
                    if observation is None:
                        observation, status = child_io_state.step(
                            action_name=temp_now_node.description, action_input=function_input)
                        call_valid = True
                    else:
                        call_valid = False
                        temp_now_node.is_loaded_from_cache = True
                        print('Returned result of function %s is loaded from cache.' % function_name)
                
                temp_node.observation = observation
                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                temp_now_node = temp_node
                # on_tool_end
                [callback.on_tool_end(
                    depth=now_depth,
                    output=observation,
                    status=status
                ) for callback in self.callbacks]
                # Check if the returned result contains any valuable information
                # Skip checking if the function name is not in the list (hallucination)
                if function_name == 'Finish':
                    print('Skip checking validity:', function_name, 'is Finish')
                elif not call_valid:
                    print('Skip checking validity since the called API is blacklisted or does not exist or the returned result is loaded from cache')
                else:
                    if self.use_blacklist:
                        input_description = self.llm.conversation_history[1]['content'].strip().split('\n')[0]
                        for f in self.io_func.functions:
                            if f['name'] == function_name:
                                api_documentation = json.dumps(f, ensure_ascii=False)
                                break
                        input_sequence = [input_description, api_documentation, function_input, observation]
                        input_sequence = [standardize(x) for x in input_sequence]
                        judgment = get_judgment(input_sequence, self.judgment_model, self.tokenizer)
                        if judgment == 1:
                            self.count_success[function_name] += 1
                            cache_valid = True
                        else:
                            self.count_failed[function_name] += 1
                            cache_valid = False
                            if (self.count_success[function_name] == 0) and (self.count_failed[function_name] >= self.th_blacklist):
                                self.remove_function(function_name, STATE_USELESS, temp_now_node)
                                print('Now blacklist function %s, remaining functions:' % function_name, [f['name'] for f in self.io_func.functions])
                    elif self.use_llm_blacklist:
                        old_messages = deepcopy(self.llm.conversation_history)
                        judgment_messages = self.get_judgment_messages(old_messages, function_name, function_input, observation)
                        self.llm.change_messages(judgment_messages)
                        judgment_response, judgment_error_code, total_tokens = self.llm.parse(self.judgment_function, process_id=self.process_id)
                        print('judgment:', judgment_response)
                        self.total_tokens += total_tokens
                        try:
                            judgment_arguments = json.loads(judgment_response['function_call']['arguments'])
                            judgment = judgment_arguments['judgment']
                            if judgment == 'Yes':
                                self.count_success[function_name] += 1
                                cache_valid = True
                            else:
                                assert(judgment == 'No')
                                self.count_failed[function_name] += 1
                                cache_valid = False
                                if (self.count_success[function_name] == 0) and (self.count_failed[function_name] >= self.th_blacklist):
                                    self.remove_function(function_name, STATE_USELESS, temp_now_node)
                                    print('Now blacklist function %s, remaining functions:' % function_name, [f['name'] for f in self.io_func.functions])
                        except Exception as e:
                            print('Judgment Error:', e)
                            cache_valid = False
                    else:
                        cache_valid = True
                    self.write_to_cache(function_name, function_input, observation, cache_valid)
                    fcost = self.io_func.cost[function_name]
                    self.budget -= fcost
                    self.remaining_budget -= fcost
                    self.io_func.budget_limit[function_name] -= 1
                    print('Function %s called successfully with cost %d. Remaining budget: %d. Remaining count: %s' % (function_name, fcost, self.budget, self.io_func.budget_limit))
                    removal = []
                    for f in self.io_func.functions:
                        fname = f['name']
                        fcost = self.io_func.cost[fname]
                        if self.io_func.budget_limit[fname] <= 0:
                            print('Function %s forbidden since its visit times exceeded' % fname)
                            removal.append(fname)
                        elif fcost > self.budget:
                            print('Function %s forbidden since its cost %d exceeds the remaining budget' % (fname, fcost))
                            removal.append(fname)
                    if len(removal) > 0:
                        for fname in removal:
                            self.remove_function(fname, STATE_EXCEEDED, temp_now_node)
                            fcost = self.io_func.cost[fname]
                        print('Remaining functions:', [f['name'] for f in self.io_func.functions])

                    if self.use_prompt:
                        self.update_cost_message(temp_now_node)

                if status != 0:
                    # return code defination can be seen in Downstream_tasks/rapid_api
                    if status == 4:
                        temp_now_node.pruned = True
                    # elif status == 1:  # hallucination api name
                    #     assert "function_call" in new_message.keys()
                    #     new_message["function_call"]["name"] = "invalid_hallucination_function_name"
                    elif status == 3:  # final answer
                        temp_now_node.is_terminal = True
                        temp_now_node.make_finish(final_answer_back_length)

            temp_now_node.messages.append(new_message)
            if temp_now_node.node_type == "Action Input":
                temp_now_node.messages.append({
                    "role": "function",
                    "name": new_message["function_call"]["name"],
                    "content": temp_now_node.observation,
                })
            return_value = None
            if not with_filter:  # DFSDT
                result = self.DFS(temp_now_node, single_chain_max_step,
                                  tree_beam_size, max_query_count, answer, with_filter)
                if len(self.terminal_node) >= answer:
                    return_value = 10000
                elif result > 1:
                    return_value = result-1

            else:

                next_tree_split_nodes.append(temp_now_node)
            self.send_agent_chain_end(
                now_depth, agent_block_ids, chain_block_ids)
            if return_value is not None:
                return return_value

        # Sort the generated next_tree_split_nodes nodes when normal DFS
        if len(next_tree_split_nodes) > 1:
            # When using normal DFS, if we have many child nodes, we will refer to LLM to compare and choose the best one to expand first
            # remember, this operator will cost extra OpenAI calls.
            LLM_rank_args = {
                "functions": self.io_func.functions,
                "process_id": self.process_id,
                "task_description": self.io_func.task_description,
                "rank_func": rank2_subfix,
            }
            scores, rank_query_count, total_tokens = sum_based_rankn(
                self.llm, LLM_rank_args=LLM_rank_args, candidates=next_tree_split_nodes)
            self.query_count += rank_query_count
            self.total_tokens += total_tokens
            for score, node in zip(scores, next_tree_split_nodes):
                node.prior_score = score
            zip_value = list(
                zip(next_tree_split_nodes, range(len(next_tree_split_nodes))))
            zip_value.sort(
                key=lambda x: x[0].prior_score, reverse=True)
            next_tree_split_nodes, filtered_order = zip(*zip_value)
            # if self.process_id == 0:
            #     print(f"score={scores}, filtered order: {filtered_order}")

        '''
        Choose one to expand
        '''
        for i in range(len(next_tree_split_nodes)):
            result = self.DFS(
                next_tree_split_nodes[i], single_chain_max_step, tree_beam_size, max_query_count, answer)
            if len(self.terminal_node) >= answer:
                return 10000
            elif result > 1:
                now_node.make_finish(2)
                return result - 1

        return 1
