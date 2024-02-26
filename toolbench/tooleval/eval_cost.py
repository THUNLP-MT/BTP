import json
import os
import re
import sys

method = sys.argv[1]
subset = sys.argv[2]

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

SENT_BLACKLIST = 'This API is forbidden since they cannot provide valuable information or usually return error messages.'
SENT_BUDGET = 'This API is forbidden since using it will exceed the budget limitation.'

def dfs(node, func):
    S = 0
    S2 = 0
    if node['node_type'] == 'Action':
        name = node['description']
        flag = False
        if name in cost:
            flag = True
        if not ('is_loaded_from_cache' in node):
            l = False
        else:
            l = node['is_loaded_from_cache']
        if l:
            flag = False
        if SENT_BLACKLIST in node['children'][0]['observation']:
            flag = False
        if SENT_BUDGET in node['children'][0]['observation']:
            flag = False
        if flag:
            S += cost[name]
            S2 += 1
    for child in node['children']:
        s, s2 = dfs(child, func)
        S += s
        S2 += s2
    return S, S2

JJ = json.load(open('../../inst/%s/%s.json' % (method, subset), 'r', encoding='utf-8'))
costs = {}
for x in JJ:
    query_id = x['query_id']
    costs[query_id] = {}
    for y in x['api_list']:
        tool_name = y['tool_name']
        api_name = y['api_name']
        final_name = '%s_for_%s' % (change_name(standardize(api_name)), standardize(tool_name))
        final_name = final_name[-64 : ]
        __cost = y['cost']
        costs[query_id][final_name] = __cost

P = json.load(open('pass_rate_results/%s_%s.json' % (subset, method), 'r', encoding='utf-8'))
Q = json.load(open('../../data2/solvable_results/%s.json' % subset, 'r', encoding='utf-8'))

total_S = 0
total_S2 = 0
exceed_cost = 0
num_pass = 0
num_pass_total = 0
eval_dir = '../../data/answer/%s/%s' % (method, subset)
for f in os.listdir(eval_dir):
    fin = open('%s/%s' % (eval_dir, f), 'r', encoding='utf-8')
    J = json.loads(fin.read())
    fin.close()
    func = set()
    for x in J['answer_generation']['function']:
        name = x['name']
        if name != 'Finish':
            func.add(name)
    root = J['tree']['tree']
    query_id = int(f.split('_')[0])
    is_solved = P[str(query_id)]['passed']
    is_failed = P[str(query_id)]['failed']
    if is_solved > is_failed:
        solved = 1
    else:
        solved = 0
    answer_chain = P[str(query_id)]['answer_steps'][8 : ]
    final_step = P[str(query_id)]['final_step'][12 : ]
    if answer_chain == final_step:
        solved = 0
    solvable_1 = Q[str(query_id)]['passed']
    unsolvable_1 = Q[str(query_id)]['failed']
    if solvable_1 > unsolvable_1:
        solvable = False
    else:
        solvable = True
    if not solvable:
        solved = 1
    cost = costs[query_id]
    total_cost, total_num = dfs(root, func)
    total_S += total_cost
    total_S2 += total_num
    if total_cost > 20:
        exceed_cost += 1
    else:
        num_pass += solved
    num_pass_total += solved

print('AC', float(total_S) / len(JJ))
print('RFBC', float(exceed_cost) / len(JJ))
print('PBC', float(num_pass) / len(JJ))
print('PR', float(num_pass_total) / len(JJ))
