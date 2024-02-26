import copy
import json
import math
import numpy as np
import re
import sys

th = 0.15

T = 1.0
C = 0.5

def sim(x, y):
    sims = float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return math.exp((sims - 1.0) / T)

emb = {}
fin = open('data2/queries_emb.jsonl', 'r', encoding='utf-8')
for x in fin:
    J = json.loads(x)
    emb[J['query']] = np.array(J['emb'])
fin.close()

fin = open('data2/merged_data_emb.jsonl', 'r', encoding='utf-8')
for x in fin:
    J = json.loads(x)
    emb[J['query']] = np.array(J['emb'])
fin.close()

fin = open('data2/retrieval_results.json', 'r', encoding='utf-8')
J = json.loads(fin.read())
fin.close()

test_queries = [x['query'] for x in J]
test_queries = set(test_queries)

fin = open('data2/merged_data.json', 'r', encoding='utf-8')
V = json.loads(fin.read())
fin.close()

valid = {}

cnt = 0
for x in V:
    query = x['query']
    if query in test_queries:
        cnt += 1
        continue
    doc = json.loads(x['doc'])
    func_name = doc['name']
    if not (func_name in valid):
        valid[func_name] = []
    valid[func_name].append((query, x['pred']))
print(cnt)

query_count = {}

fin = open('data2/count_calls.jsonl', 'r', encoding='utf-8')
for x in fin:
    X = json.loads(x)
    func_name = X['function_name']
    if not (func_name in query_count):
        query_count[func_name] = []
    cnt = X['count']
    if X['pred'] == 0:
        cnt = min(cnt, 1)
    query_count[func_name].append((X['query'], cnt))
fin.close()

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

JJ = []
for x in J:
    y = copy.deepcopy(x)
    n_cands = len(y['candidates'])
    for i in range(0, n_cands):
        doc = y['candidates'][i]
        J_doc = json.loads(doc)
        tool_name = J_doc['tool_name']
        api_name = J_doc['api_name']
        final_name = '%s_for_%s' % (change_name(standardize(api_name)), standardize(tool_name))
        final_name = final_name[-64 : ]
        y['candidates'][i] = {}
        y['candidates'][i]['doc'] = doc
        y['candidates'][i]['func_name'] = final_name
        pred_1 = []
        pred_0 = []
        sim_total = 0.0
        sim_valid = 0.0
        query = y['query']
        if final_name in valid:
            for q, pred in valid[final_name]:
                sims = sim(emb[q], emb[query])
                sim_total += sims
                sim_valid += sims * pred
        n_budget = 0.0
        n_queries = 0.0
        if final_name in query_count:
            for q, budget in query_count[final_name]:
                sims = sim(emb[q], emb[query])
                n_budget += sims * budget
                n_queries += sims
        y['candidates'][i]['sim_total'] = sim_total
        y['candidates'][i]['sim_valid'] = sim_valid

        score = (sim_valid + 0.1) / (sim_total + 1.0)
        y['candidates'][i]['score'] = score

        if abs(n_queries) < 1e-5:
            y['candidates'][i]['budget'] = 1
        elif score < float(th):
            y['candidates'][i]['budget'] = 0
        else:
            y['candidates'][i]['budget'] = int(round(n_budget / n_queries, 0))
    JJ.append(y)

fout = open('filtered_results_count_th%s.json' % th, 'w', encoding='utf-8')
fout.write(json.dumps(JJ, ensure_ascii=False, indent=4))
fout.close()
