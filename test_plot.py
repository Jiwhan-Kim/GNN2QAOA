import matplotlib.pyplot as plt
import numpy as np

from t1 import *

x = np.arange(55)
no_model = np.array(no_model6)
model = np.array(model6)

plt.bar(x - 0.2, no_model, width=0.4, label='No Model')
plt.bar(x + 0.2, model, width=0.4, label='Model')
plt.xticks(x)
plt.legend()
plt.tight_layout()

plt.xlabel('Graph')
plt.ylabel('Cost')
plt.show()

sub = model - no_model
cnt = np.sum(sub > 0)
print(cnt)
