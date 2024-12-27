initial_table = {
    "Burglary": {True: 0.001, False: 0.999},
    "Earthquake": {True: 0.002, False: 0.998},
    "Alarm": {
        # (B, E): {A}
        (True, True): {True: 0.95, False: 0.05},
        (True, False): {True: 0.94, False: 0.06},
        (False, True): {True: 0.29, False: 0.71},
        (False, False): {True: 0.001, False: 0.999},
    },
    "JohnCalls": {
        # (A): {J}
        True: {True: 0.9, False: 0.1},
        False: {True: 0.05, False: 0.95},
    },
    "MaryCalls": {
        # (A): {M}
        True: {True: 0.7, False: 0.3},
        False: {True: 0.01, False: 0.99},
    }
}

def joint_distribution(table):
    joint_table = {}
    for b in range(2):
        for e in range(2):
            for a in range(2):
                for j in range(2):
                    for m in range(2):
                        joint_table[(b, e, a, j, m)] = (table["Burglary"][b] * table["Earthquake"][e] * table["Alarm"][(b,e)][a] * table["JohnCalls"][a][j] * table["MaryCalls"][a][m])
    return joint_table

def variable_elimination(joint_table, query, evidence):         
    eliminated_summed_up = {}
      
    # if no evidence
    if evidence == {}:
        eliminated_summed_up = {query: {True: 0, False: 0}}

        for key in joint_table:
            b, e, a, j, m = key

            if (query == "Burglary" and b == 1):
                eliminated_summed_up[query][True] += joint_table[key]
            elif (query == "Earthquake" and e == 1):
                eliminated_summed_up[query][True] += joint_table[key]
            elif (query == "Alarm" and a == 1):
                eliminated_summed_up[query][True] += joint_table[key]
            elif (query == "JohnCalls" and j == 1):
                eliminated_summed_up[query][True] += joint_table[key]
            elif (query == "MaryCalls" and m == 1):
                eliminated_summed_up[query][True] += joint_table[key]

        eliminated_summed_up[query][False] = 1 - eliminated_summed_up[query][True]

    # else - if there are evidences
    else:
        eliminated = {}
        factor = 0
        # first, eliminate the variables - keep the ones that matter only
        for evidence_key, evidence_val in evidence.items():
            for key in joint_table:
                b, e, a, j, m = key

                # if evidence is from the child most nodes
                if evidence_key == "JohnCalls" and j == evidence_val:

                    # factor from the probs where key == j
                    factor += joint_table[key]
                    
                    # objective query from parent most nodes
                    if query == "Burglary":
                        if (b, j) not in eliminated:
                            eliminated[(b, j)] = joint_table[key]
                        else:
                            eliminated[(b, j)] += joint_table[key]
                    elif query == "Earthquake":
                        if (e, j) not in eliminated:
                            eliminated[(e, j)] = joint_table[key]
                        else:
                            eliminated[(e, j)] += joint_table[key]
                    elif query == "Alarm":
                        if (a, j) not in eliminated:
                            eliminated[(a, j)] = joint_table[key]
                        else:
                            eliminated[(a, j)] += joint_table[key]
                    elif query == "MaryCalls":
                        if (m, j) not in eliminated:
                            eliminated[(m, j)] = joint_table[key]
                        else:
                            eliminated[(m, j)] += joint_table[key]
                        
                elif evidence_key == "MaryCalls" and m == evidence_val:
                    factor += joint_table[key]

                    if query == "Burglary":
                        if (b, a) not in eliminated:
                            eliminated[(b, m)] = joint_table[key]
                        else:
                            eliminated[(b, m)] += joint_table[key]
                    elif query == "Earthquake":
                        if (e, m) not in eliminated:
                            eliminated[(e, m)] = joint_table[key]
                        else:
                            eliminated[(e, m)] += joint_table[key]
                    elif query == "Alarm":
                        if (a, m) not in eliminated:
                            eliminated[(a, m)] = joint_table[key]
                        else:
                            eliminated[(a, m)] += joint_table[key]
                    elif query == "JohnCalls":
                        if (j, m) not in eliminated:
                            eliminated[(j, m)] = joint_table[key]
                        else:
                            eliminated[(j, m)] += joint_table[key]
                
                elif evidence_key == "Alarm" and a == evidence_val:
                    factor += joint_table[key]

                    if query == "Burglary":
                        if (b, a) not in eliminated:
                            eliminated[(b, a)] = joint_table[key]
                        else:
                            eliminated[(b, a)] += joint_table[key]
                    elif query == "Earthquake":
                        if (e, a) not in eliminated:
                            eliminated[(e, a)] = joint_table[key]
                        else:
                            eliminated[(e, a)] += joint_table[key]
                    elif query == "JohnCalls":
                        if (a, j) not in eliminated:
                            eliminated[(a, j)] = joint_table[key]
                        else:
                            eliminated[(a, j)] += joint_table[key]
                    elif query == "MaryCalls":
                        if (a, m) not in eliminated:
                            eliminated[(a, m)] = joint_table[key]
                        else:
                            eliminated[(a, m)] += joint_table[key]
                    
                elif evidence_key == "Burglary" and b == evidence_val:
                    factor += joint_table[key]

                    if query == "Earthquake":
                        if (b, e) not in eliminated:
                            eliminated[(b, e)] = joint_table[key]
                        else:
                            eliminated[(b, e)] += joint_table[key]
                    elif query == "Alarm":
                        if (b, a) not in eliminated:
                            eliminated[(b, a)] = joint_table[key]
                        else:
                            eliminated[(b, a)] += joint_table[key]
                    elif query == "JohnCalls":
                        if (b, j) not in eliminated:
                            eliminated[(b, j)] = joint_table[key]
                        else:
                            eliminated[(b, j)] += joint_table[key]
                    elif query == "MaryCalls":
                        if (b, m) not in eliminated:
                            eliminated[(b, m)] = joint_table[key]
                        else:
                            eliminated[(b, m)] += joint_table[key]
                    
                elif evidence_key == "Earthquake" and e == evidence_val:
                    factor += joint_table[key]

                    if query == "Burglary":
                        if (b, e) not in eliminated:
                            eliminated[(b, e)] = joint_table[key]
                        else:
                            eliminated[(b, e)] += joint_table[key]
                    elif query == "Alarm":
                        if (e, a) not in eliminated:
                            eliminated[(e, a)] = joint_table[key]
                        else:
                            eliminated[(e, a)] += joint_table[key]
                    elif query == "JohnCalls":
                        if (e, j) not in eliminated:
                            eliminated[(e, j)] = joint_table[key]
                        else:
                            eliminated[(e, j)] += joint_table[key]
                    elif query == "MaryCalls":
                        if (e, m) not in eliminated:
                            eliminated[(e, m)] = joint_table[key]
                        else:
                            eliminated[(e, m)] += joint_table[key]

        # second, add up the eliminated probabilities according to the query
        eliminated_summed_up = {0: 0, 1: 0}
        for key in eliminated:
            eliminated_summed_up[key[0]] += eliminated[key]
        
        # lastly, normalize the resulting table
        for key in eliminated_summed_up:
            eliminated_summed_up[key] = eliminated_summed_up[key] / factor
    
    return eliminated_summed_up

# algorithm application
query = "Burglary"
evidence_key = "JohnCalls"
evidence_val = True
final_table = variable_elimination(joint_distribution(initial_table), query, {evidence_key: evidence_val})
# print(final_table)

# print the final table
for query_state, probability in final_table.items():
    if query_state == 0:
        print(f"P({query} = False | {evidence_key} = {evidence_val}) = {probability}")
    else:
        print(f"P({query} = True | {evidence_key} = {evidence_val}) = {probability}")
