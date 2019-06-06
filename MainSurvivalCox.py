from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import pandas as pd

regression_dataset = pd.read_csv("C:/Users/Josem/Dropbox/PHD/Proyectos/2019-05 - BRCA/genetic_yes.csv", delimiter="\t")

regression_dataset['Class'] = regression_dataset['Class'].map({'YES': 1, 'NO': 0})
regression_dataset = regression_dataset.join(pd.get_dummies(regression_dataset['clusters']))
regression_dataset = regression_dataset.drop(['id', 'clusters'], axis=1)

print(regression_dataset.head(100))
print(regression_dataset.dtypes)

kmf = KaplanMeierFitter()
kmf.fit(regression_dataset['Time'], event_observed=regression_dataset['Class'])

kmf.survival_function_
survival_function = kmf.plot_survival_function()  # or just kmf.plot()
survival_function.get_figure().savefig("KaplanMeierFitter.png")

# Using Cox Proportional Hazards model
cph = CoxPHFitter(penalizer=0.1)
cph.fit(regression_dataset, 'Time', event_col='Class')
cph.print_summary()

# predict=cph.predict_survival_function(regression_dataset).plot()
# predict.get_figure().savefig("predicted.png")

ax = cph.plot()
ax.get_figure().savefig("CoxPHFitter.png")
