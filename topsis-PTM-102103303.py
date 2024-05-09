#Chirag Singla - 102103303 - COE11
import numpy as np


decision_matrix = np.array([
    [0.85, 0.75, 0.65],  # Model A
    [0.80, 0.70, 0.70],  # Model B
    [0.90, 0.80, 0.60]   # Model C
])


normalized_matrix = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))


weights = np.array([0.4, 0.3, 0.3])


weighted_normalized_matrix = normalized_matrix * weights


ideal_solution = np.max(weighted_normalized_matrix, axis=0)
negative_ideal_solution = np.min(weighted_normalized_matrix, axis=0)


distance_to_ideal = np.sqrt(((weighted_normalized_matrix - ideal_solution) ** 2).sum(axis=1))
distance_to_negative_ideal = np.sqrt(((weighted_normalized_matrix - negative_ideal_solution) ** 2).sum(axis=1))


relative_closeness = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)


ranked_indices = np.argsort(relative_closeness)[::-1]


print("Ranking of Models:")
for rank, idx in enumerate(ranked_indices):
    print(f"Rank {rank + 1}: Model {chr(65 + idx)}")


import matplotlib.pyplot as plt


plt.figure(figsize=(8, 5))
plt.imshow(decision_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Performance')
plt.title('Decision Matrix')
plt.xlabel('Criteria')
plt.ylabel('Models')
plt.xticks(np.arange(3), ['Accuracy', 'Fluency', 'Diversity'])
plt.yticks(np.arange(3), ['Model A', 'Model B', 'Model C'])
plt.show()
