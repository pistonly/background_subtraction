import pandas as pd


methods = []
fps = []
with open("../results/method_vs_fps.txt") as f:
    while True:
        line = f.readline()
        if line:
            if "<class" in line:
                method = line.split("pybgs.")[-1].split("'")[0]
                methods.append(method)
            if "fps:" in line:
                fps_i = float(line.split(":")[-1])
                fps.append(fps_i)
        else:
            break

method_fps = [[m, f] for m, f in zip(methods, fps)]
df = pd.DataFrame(method_fps, columns=['method', 'fps'])
df = df.sort_values('fps', ascending=False)
df.index = range(len(df))
print(df)
for m, f in zip(methods, fps):
    print(f"{m}: {f}")
