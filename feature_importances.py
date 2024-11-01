import shap
import json
import numpy as np

class FeatureImportance:
    def fetch_shap_values(model, x_train, x_test):
        shap_explainer = shap.Explainer(model.infer, x_train)
        shap_values = shap_explainer(x_test)
        return shap_values
    
    def save_shap_values(shap_values, out):
        values = np.abs(shap_values.values).mean(axis = 0)
        val_dict = dict()
        for i,val in enumerate(values):
          val_dict[i] = val
        with open(out, 'w') as f:
            json.dump(val_dict, f)