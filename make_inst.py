import json
import os
import re
import sys

th = 0.15

scores = {}
budgets = {}

B = json.load(open('filtered_results_count_th%s.json' % th, 'r', encoding='utf-8'))
for x in B:
    query = x['query']
    for func in x['candidates']:
        func_name = func['func_name']
        comb = query, func_name
        scores[comb] = func['score']
        budgets[comb] = func['budget']

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

max_cost = 20

os.system('mkdir -p inst/ours_cost%d_th%s' % (max_cost, th))

query_types = ['G1_instruction', 'G1_tool', 'G1_category', 'G2_instruction', 'G2_category', 'G3_instruction']
for query_type in query_types:
    J = json.load(open('data2/inst_with_cost/%s.json' % query_type, 'r', encoding='utf-8'))
    J1 = []
    for x in J:
        cost = []
        value = []
        idx = []
        query = x['query']
        for i in range(0, len(x['api_list'])):
            tool_name = x['api_list'][i]['tool_name']
            api_name = x['api_list'][i]['api_name']
            final_name = '%s_for_%s' % (change_name(standardize(api_name)), standardize(tool_name))
            final_name = final_name[-64 : ]
            score = scores[(query, final_name)]
            budget = budgets[(query, final_name)]
            cost += [x['api_list'][i]['cost']] * budget
            value += [score] * budget
            idx += [i] * budget
            x['api_list'][i]['score'] = score
            x['api_list'][i]['budget_limit'] = 0
        f = [[0.0] * (max_cost + 1)]
        g = [[0] * (max_cost + 1)]
        for i in range(0, len(cost)):
            f.append([])
            g.append([])
            for j in range(0, max_cost + 1):
                f[-1].append(f[-2][j])
                g[-1].append(0)
                if j >= cost[i]:
                    new_value = f[-2][j - cost[i]] + value[i]
                    if new_value > f[-1][j]:
                        f[-1][j] = new_value
                        g[-1][j] = cost[i]
        dest = 0
        for i in range(1, max_cost + 1):
            if f[-1][i] > f[-1][dest]:
                dest = i
        for i in range(len(cost), 0, -1):
            if g[i][dest] > 0:
                x['api_list'][idx[i - 1]]['budget_limit'] += 1
            dest -= g[i][dest]
        J1.append(x)
    json.dump(J1, open('inst/ours_cost%d_th%s/%s.json' % (max_cost, th, query_type), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

