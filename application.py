import pandas as pd
import numpy as np
import os
from tensorflow import keras
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

## AIF360 - A library for fairness assessments
from aif360.datasets import BinaryLabelDataset  # To handle the data
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric



def calculate_fairness_metrics(file_path):  #grouping_variable
    # Perform calculations for fairness metrics
    # For example, you can calculate the fairness metrics as in your original code
    # print("Load npy:")
    try:
        df1 = pd.read_csv(file_path)[
        ["patient_id", "age", "sex", "weight", "height"]]


        a = np.load("data/patients_train.npy")
        b = np.load("data/patients_val.npy")
        c = np.load("data/patients_test.npy")


        X_train = np.load("data/X_train.npy")
        X_test = np.load("data/X_test.npy")
        X_val = np.load("data/X_val.npy")

        y_train = np.load("data/y_train.npy")
        y_test = np.load("data/y_test.npy")
        y_val = np.load("data/y_val.npy")


        # y_val

        # print("pd Dataframe:")
        df = pd.DataFrame(columns=["data_index", "Partition", "patient_id"])

        X = np.concatenate([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])

        partitions = np.concatenate(
            [
                np.array(["Train"] * len(X_train)),
                np.array(["Validation"] * len(X_val)),
                np.array(["Test"] * len(X_test)),
            ]
        )

        patient_ids = np.concatenate([a, b, c])

        _X = np.concatenate([X_train, X_val, X_test])
        _y = np.concatenate([y_train, y_val, y_test])
        indexes = np.concatenate(
            [
                np.arange(0, len(X_train), 1),
                np.arange(0, len(X_val), 1),
                np.arange(0, len(X_test), 1),
            ]
        )

        # df["X"] = _X
        df["y"] = _y

        df["data_index"] = indexes

        df["Partition"] = partitions
        df["patient_id"] = patient_ids

        df["patient_id"] = df["patient_id"].astype(int)

        # joint_dfs = df.join(df1, lsuffix='patient_id', rsuffix='patient_id')
        joint_dfs = pd.merge(df, df1, on="patient_id")


        DATA_DIR = "./data"


        file_names = [
            "X_train.npy",
            "y_train.npy",
            "X_val.npy",
            "y_val.npy",
            "X_test.npy",
            "y_test.npy",
            "patients_train.npy",
            "patients_val.npy",
            "patients_test.npy",
        ]

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,  ## 2 X_test
            y_test,  ## 3 y_test
            patients_train,
            patients_val,
            patients_test,
        ) = [np.load(os.path.join(DATA_DIR, f), allow_pickle=True) for f in file_names]

        # load ptbx CSV file to access demographic data based on patient ids
        # print("Read CSV:")
        df_ptbxl = pd.read_csv(os.path.join(os.path.dirname(DATA_DIR), file_path))

        # load model
        # print("Before model architecture:")
        model = keras.models.load_model(os.path.join(DATA_DIR, "model.keras"))   ## 2 model
        # print("Summary of model architecture:\n")
        print(model.summary())



        def get_patient_data(patient_id: int):
            """
            Get all available demographic data for a given patient id
            """
            df_patients = df_ptbxl.loc[df_ptbxl["patient_id"] == patient_id]

            if len(df_patients) == 0:
                raise Exception(f'No patient found with id "{patient_id}"')

            # we handle this case by using the first ECG signal provided for this patient
            if len(df_patients) > 1:
                print(f'\nWARNING: Found multiple patients with id "{patient_id}"\n')

            return df_patients.iloc[0]


        def evaluate_model(model, X_test, y_test):  
            if X_test.ndim == 2:
                X_test = np.expand_dims(X_test, axis=0)

            y_pred = model.predict(X_test, verbose=0).flatten()

            # convert to integer labels
            y_pred_binary = (y_pred >= 0.5).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary, labels=[0, 1]).ravel()

            # What proportion of positive identifications was actually correct?
            precision = tp / (tp + fp)
            # What proportion of actual positives was identified correctly?
            recall = tp / (tp + fn)
            f1 = 2.0 * (precision * recall) / (precision + recall)
            accuracy = (tp + tn) / (tp + fn + fp + tn)

            roc_auc = roc_auc_score(y_test, y_pred)

            df_report = pd.DataFrame(
                data={
                    "accuracy": round(accuracy, 4),
                    "auc": round(roc_auc, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                },
                index=[0],
            )

            return df_report


        # just used to illustrate usage of classifier
        for i in range(0, 10):
            x = X_test[i]
            y = y_test[i]
            patient_id = patients_test[i]

            x_expanded = np.expand_dims(x, axis=0)

            pred = model.predict(x_expanded, verbose=0)
            classification_score = pred[0][0]
            pred_class = "MI" if classification_score >= 0.5 else "non-MI"

            patient = get_patient_data(patient_id)

            print(f"Test data point {i} from patient {patient_id}:")
            print(f"y_true: {y}")
            print(f"y_pred: {classification_score}")
            print(f"class: {pred_class}")
            print("patient data: \n")
            print(patient)
            print("-" * 75)


        df_model_report = evaluate_model(model, X_test, y_test)

        print(f"\nModel Evaluation ({len(y_test)} test samples):")

        age_median = df_ptbxl["age"].median()
        age_median


        # get_patient_data(15709)["sex"]

        female_index = []
        male_index = []

        younger_index = []
        older_index = []
        gender_groupings = []
        age_groupings = []


        for i in range(0, len(y_test)):
            patient_id = patients_test[i]
            patient = get_patient_data(patient_id)

            if patient["sex"] == 0:
                male_index.append(i)
                gender_groupings.append(0)
            else:
                female_index.append(i)
                gender_groupings.append(1)

            if patient["age"] > age_median:
                older_index.append(i)
                age_groupings.append(1)
            else:
                younger_index.append(i)
                age_groupings.append(0)

        female_index = np.array(female_index)
        male_index = np.array(male_index)
        older_index = np.array(older_index)
        younger_index = np.array(younger_index)

        gender_groupings = np.array(gender_groupings) ## 2 gender_groupings
        age_groupings = np.array(age_groupings)


        X_test_female = X_test[female_index]
        X_test_male = X_test[male_index]

        y_test_female = y_test[female_index]
        y_test_male = y_test[male_index]

        X_test_young = X_test[younger_index]
        X_test_old = X_test[older_index]

        y_test_young = y_test[younger_index]
        y_test_old = y_test[older_index]


        def custom_model_eval(model, _X, _y):
            if _X.ndim == 2:
                _X = np.expand_dims(_X, axis=0)

            y_pred = model.predict(_X, verbose=0).flatten()

            # convert to integer labels
            y_pred_binary = (y_pred >= 0.5).astype(int)

            return y_pred_binary


        y_pred_all = custom_model_eval(model, X_test, y_test) 

        class SPATIALFairnessModule:
            ## Assumes binary values (1,0) in all the inputs, we can add extra parameters if that's not the case
            def __init__(self, _y_true, _y_pred, _groups, _group_name):
                self.y_true = _y_true  # Binary array. Contains the true label for each sample
                self.y_pred = _y_pred  # Binary array. Contains the predictions for each sample
                self.groups = _groups  # Binary array. Defines the demogaphic group membership of each sample
                self.grouping_name = (
                    _group_name  # String. Gives a name to the grouping. Example: Gender
                )

                _label_names = ["Y"]
                _protected_attribute_names = [_group_name]
                _favorable_label = 1
                _unfavorable_label = 0

                _df = pd.DataFrame(columns=["Y", _group_name])
                _df["Y"] = _y_true
                _df[_group_name] = _groups

                self.aif_input_data = BinaryLabelDataset(
                    df=_df,
                    label_names=_label_names,
                    protected_attribute_names=_protected_attribute_names,
                    favorable_label=_favorable_label,
                    unfavorable_label=_unfavorable_label,
                )

                self.privileged_groups = [{_group_name: 1}]
                self.unprivileged_groups = [{_group_name: 0}]

                self.input_metrics = BinaryLabelDatasetMetric(
                    self.aif_input_data,
                    unprivileged_groups=self.unprivileged_groups,
                    privileged_groups=self.privileged_groups,
                )

                self.pred_aif_data = self.aif_input_data.copy(deepcopy=True)
                self.pred_aif_data.labels = _y_pred

                self.clf_metrics = ClassificationMetric(
                    self.aif_input_data,
                    self.pred_aif_data,
                    unprivileged_groups=self.unprivileged_groups,
                    privileged_groups=self.privileged_groups,
                )

            ##########################################
            # PRE-TRAINING BIAS: BIAS IN INPUT DATA
            ##########################################

            ## Consistency -> Individual Fairness approach
            ## Measures how similar the labels are for similar instances
            # 0 is the desirable value
            def get_consistency_input(self):
                return self.input_metrics.consistency()[0]

            ## Class imbalance
            ## Compares ratio of positives among the classess
            # 0 is the desirable value
            def get_class_imbalance_input(self):
                ## Num positives between privileged samples
                np = self.input_metrics.num_positives(privileged=True)
                ## Num positives between unprivileged samples
                nd = self.input_metrics.num_positives(privileged=False)

                ## Class imbalance
                ci = (np - nd) / (nd + np)

                return ci

            ## Disparate impact is equivalent to Statistical parity difference
            ## Measures differences in selection rates.
            ## O is desirable value
            def get_disparate_impact_input(self):
                di = self.input_metrics.statistical_parity_difference()

                return di

            ##########################################
            # POST-TRAINING BIAS: BIAS IN PREDICTIONS
            ##########################################
            ## Disparate impact is equivalent to Statistical parity difference
            def get_disparate_impact_prediction(self):
                di = self.clf_metrics.statistical_parity_difference()

                return di

            def get_equal_oportunity_prediction(self):
                eq_op = self.clf_metrics.equal_opportunity_difference()

                return eq_op

            def get_equalized_odds(self):
                avg_odds = self.clf_metrics.average_odds_difference()

                return avg_odds

            #########################################
            # VISUALIZATION AND COMPLETE ANALYSIS
            #########################################

            # Converts a fairness continuous value into a care label category
            # Inspired from: https://www.frontiersin.org/articles/10.3389/frai.2022.975029/full
            def get_score_care_label(self, _v):
                _v = abs(_v)

                if _v <= 0.05:
                    return "A"
                elif _v <= 0.25:
                    return "B"
                elif _v <= 0.5:
                    return "C"
                elif _v <= 0.75:
                    return "D"
                else:
                    return "E"

            # The model is as unfair as the worst fairness metric reports
            def get_predictions_fairness_score(self):
                worst_score = 0
                worst_metric = ""

                _fairness_functions = [
                    ("Disparate impact", self.get_disparate_impact_prediction),
                    ("Equal opportunity", self.get_equal_oportunity_prediction),
                    ("Equalized odds", self.get_equalized_odds),
                ]

                for f in _fairness_functions:
                    score = f[1]()

                    if abs(score) > worst_score:
                        worst_score = abs(score)
                        worst_metric = f[0]

                worst_score_Deci = "{:.3f}".format(worst_score)

                return worst_score_Deci, worst_metric, self.get_score_care_label(worst_score)


        # if grouping_variable == 'Gender':  ## 1 y_test, 1 y_pred_all , 1 gender_groupings
        fairness_for_gender = SPATIALFairnessModule(y_test, y_pred_all, gender_groupings, "Gender")
        print(" --- GENDER INPUT ---")
        consistency_Gender =  fairness_for_gender.get_consistency_input()
        consistency_Gender = "{:.3f}".format(consistency_Gender)

        class_imbalance_Gender = fairness_for_gender.get_class_imbalance_input()
        class_imbalance_Gender = "{:.3f}".format(class_imbalance_Gender)

        disparate_impact_input_Gender = fairness_for_gender.get_disparate_impact_input()
        disparate_impact_input_Gender = "{:.3f}".format(disparate_impact_input_Gender)
            
        print(" --- GENDER PREDICTION ---")
        disparate_impact_prediction_Gender =  fairness_for_gender.get_disparate_impact_prediction()
        disparate_impact_prediction_Gender = "{:.3f}".format(disparate_impact_prediction_Gender)

        equal_opportunity_Gender = fairness_for_gender.get_equal_oportunity_prediction()
        equal_opportunity_Gender = "{:.3f}".format(equal_opportunity_Gender)
       
        equalized_odds_Gender = fairness_for_gender.get_equalized_odds()
        equalized_odds_Gender = "{:.3f}".format(equalized_odds_Gender)
        
        overall_fairness_score_Gender =  fairness_for_gender.get_predictions_fairness_score()

     
        # else:
        fairness_for_age = SPATIALFairnessModule(y_test, y_pred_all, age_groupings, "Age")
        print(" --- AGE INPUT ---")
        consistency_Age =  fairness_for_age.get_consistency_input()
        consistency_Age = "{:.3f}".format(consistency_Age)

        class_imbalance_Age = fairness_for_age.get_class_imbalance_input()
        class_imbalance_Age = "{:.3f}".format(class_imbalance_Age)

        disparate_impact_input_Age = fairness_for_age.get_disparate_impact_input()
        disparate_impact_input_Age = "{:.3f}".format(disparate_impact_input_Age)
            
        print(" --- AGE PREDICTION ---")
        disparate_impact_prediction_Age =  fairness_for_age.get_disparate_impact_prediction()
        disparate_impact_prediction_Age = "{:.3f}".format(disparate_impact_prediction_Age)

        equal_opportunity_Age = fairness_for_age.get_equal_oportunity_prediction()
        equal_opportunity_Age = "{:.3f}".format(equal_opportunity_Age)

        equalized_odds_Age = fairness_for_age.get_equalized_odds()
        equalized_odds_Age = "{:.3f}".format(equalized_odds_Age)

        overall_fairness_score_Age =  fairness_for_age.get_predictions_fairness_score()

        consistency_des = "Measures how similar the labels are for similar instances, 0 is the desirable value"
        class_imbalance_des = "Compares ratio of positives among the classess, 0 is the desirable value"
        disparate_impact_input_des = "Disparate impact is equivalent to Statistical parity difference. It measures differences in selection rates, O is desirable value"
        disparate_impact_prediction_des = "Predict the differences in selection rates"
        equal_opportunity_des = "Computes the equal opportunity difference, which measures whether the model provides equal opportunities for positive outcomes across different groups"
        equalized_odds_des = "Calculates the average odds difference, which assesses whether the model exhibits equalized odds across different groups"
        overall_fairness_score_des = "The overall fairness score aggregates various fairness metrics. The score is as bad as the worst performing metric for your model."


        results = {
            # 'grouping_variable': grouping_variable,
            'consistency_Gender': consistency_Gender,
            'class_imbalance_Gender': class_imbalance_Gender,
            'disparate_impact_input_Gender': disparate_impact_input_Gender,
            'disparate_impact_prediction_Gender': disparate_impact_prediction_Gender,
            'equal_opportunity_Gender': equal_opportunity_Gender,
            'equalized_odds_Gender': equalized_odds_Gender,
            'overall_fairness_score_Gender': overall_fairness_score_Gender,
            'consistency_Age': consistency_Age,
            'class_imbalance_Age': class_imbalance_Age,
            'disparate_impact_input_Age': disparate_impact_input_Age,
            'disparate_impact_prediction_Age': disparate_impact_prediction_Age,
            'equal_opportunity_Age': equal_opportunity_Age,
            'equalized_odds_Age': equalized_odds_Age,
            'overall_fairness_score_Age': overall_fairness_score_Age,
            'consistency_des': consistency_des,
            'class_imbalance_des': class_imbalance_des,
            'disparate_impact_input_des' : disparate_impact_input_des,
            'disparate_impact_prediction_des': disparate_impact_prediction_des,
            'equal_opportunity_des' : equal_opportunity_des,
            'equalized_odds_des': equalized_odds_des,
            'overall_fairness_score_des' : overall_fairness_score_des
        }


        return results

    except Exception as e:
        raise Exception(str(e))
