# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


y = np.array([2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1])
scores = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.5, 0.5, 0.5, 0.5, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1])
target = 2

def generate_ROC(scores, real, target):
	data = pd.DataFrame(data = {"y": real, "score": scores})
	# Ordenando scores descendientemente
	data.sort_values("score", ascending = False, inplace = True)
	# Cantidad de valores clasificados como positivos
	cm = {"TP": 0.0, "FP": 0.0}
	# Cantidad de instancias positivas y negativas 
	targetCount = data.y == target
	yValues = {"P": np.float(len(data.y[targetCount])), "N": np.float(len(data.y[-targetCount]))}

	result = pd.DataFrame(columns = ["x", "y", "score"])
	prev = np.Inf
	indices = range(len(data))

	for i in indices:
		if data.score[i] != prev:
			result.loc[len(result)] = [cm["FP"]/yValues["N"], cm["TP"]/yValues["P"], prev]
			prev = data.score[i]

		if data.y[i] == target:
			cm["TP"] = cm["TP"] + 1
		else:
			cm["FP"] = cm["FP"] + 1

	# Ultimo punto (1,1)
	result.loc[len(result)] = [cm["FP"]/yValues["N"], cm["TP"]/yValues["P"], prev]

	return result

roc = generate_ROC(scores, y, target)

# Graficando Curva ROC
plt.figure()
plt.plot([0, 1], [0, 1], "y--", label = "luck")
plt.plot(roc.x, roc.y, "k--", label = "ROC")
plt.plot(roc.x, roc.y, "ro", label = "score")
for i, txt in enumerate(roc.score):
	plt.annotate(txt, (roc.x[i],roc.y[i]))

plt.xlim([-0.008, 1.008])
plt.ylim([-0.008, 1.008])

plt.xlabel("FP-Rate")
plt.ylabel("TP-Rate")
plt.title("ROC Curve")
plt.legend(loc = "lower right")

print "Guardando la imagen"

plt.savefig("roc.png")