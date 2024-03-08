import csv

with open("result_matrix.md", "rt") as f:
    md = list(map(str.strip, f))

with open("result_matrix.csv", "wt") as f:
    w = csv.writer(f)
    w.writerow(["Self-play", "Dense reward", "Health", "Temporal actions", "Obs Play", "Obs Move Prog", "Opponent", "Actor LR", "Critic LR", "Actor H coef", "Actor HS", "Actor HA", "Critic HS", "Critic HA", "Delta", "Reward", "Length"])
    for l in md:
        vs = l[2:-2].split(" | ")
        for i, v in enumerate(vs):
            if v == "{{:green-ball}}":
                vs[i] = "T"
            elif v == "{{:red-ball}}":
                vs[i] = "F"
        w.writerow(vs)
