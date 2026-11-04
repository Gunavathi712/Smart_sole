import csv
import math
import random

random.seed(42)

X, Y = [], []
with open("ulcer_dataset.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        x = [
            float(r["FSR1"]) / 4095.0,
            float(r["FSR2"]) / 4095.0,
            float(r["FSR3"]) / 4095.0,
            float(r["FSR4"]) / 4095.0,
            float(r["FSR5"]) / 4095.0,
            float(r["TempDelta_C"]) / 4.0,
            float(r["Tilt_Deg"]) / 35.0
        ]
        X.append(x)
        Y.append(int(r["Risk_Level"]))
IN_D, HID_D, OUT_D = 7, 8, 3
W1 = [[random.uniform(-0.5, 0.5) for _ in range(IN_D)] for _ in range(HID_D)]
b1 = [0.0] * HID_D
W2 = [[random.uniform(-0.5, 0.5) for _ in range(HID_D)] for _ in range(OUT_D)]
b2 = [0.0] * OUT_D

def forward(x):
    h = [max(0.0, b1[i] + sum(W1[i][j] * x[j] for j in range(IN_D))) for i in range(HID_D)]
    logits = [b2[k] + sum(W2[k][i] * h[i] for i in range(HID_D)) for k in range(OUT_D)]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return h, [e / s for e in exps]
lr = 0.05
for epoch in range(100):
    for x, y in zip(X, Y):
        h, probs = forward(x)
        dl = [probs[k] - (1.0 if k == y else 0.0) for k in range(OUT_D)]
        dh = [sum(W2[k][i] * dl[k] for k in range(OUT_D)) if h[i] > 0 else 0.0 for i in range(HID_D)]
        for k in range(OUT_D):
            b2[k] -= lr * dl[k]
            for i in range(HID_D):
                W2[k][i] -= lr * dl[k] * h[i]
        for i in range(HID_D):
            b1[i] -= lr * dh[i]
            for j in range(IN_D):
                W1[i][j] -= lr * dh[i] * x[j]

correct = sum(1 for x, y in zip(X, Y) if forward(x)[1].index(max(forward(x)[1])) == y)
print(f"AI Training Complete! Accuracy: {100.0 * correct / len(Y):.2f}%")

with open("model_weights.h", "w") as f:
    f.write("// AUTO-GENERATED AI WEIGHTS FROM 1500 PATIENT DATASET\n\n")
    f.write("const float NN_W1[8][7] = {\n")
    for row in W1:
        f.write("  { " + ", ".join(f"{v:.4f}f" for v in row) + " },\n")
    f.write("};\n\n")
    f.write("const float NN_B1[8] = { " + ", ".join(f"{v:.4f}f" for v in b1) + " };\n\n")
    f.write("const float NN_W2[3][8] = {\n")
    for row in W2:
        f.write("  { " + ", ".join(f"{v:.4f}f" for v in row) + " },\n")
    f.write("};\n\n")
    f.write("const float NN_B2[3] = { " + ", ".join(f"{v:.4f}f" for v in b2) + " };\n")

print("SUCCESS! model_weights.h created for ESP32!")