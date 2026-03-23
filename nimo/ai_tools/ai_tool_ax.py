import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ax.service.ax_client import AxClient, ObjectiveProperties


class AX():
    """Bayesian optimisation over a discrete candidate pool using Ax / BoTorch.

    Workflow
    --------
    1. Normalise all candidate features to [0, 1].
    2. Register training observations with an AxClient GP experiment.
    3. Ask Ax to suggest the next point in the normalised feature space.
    4. Return the *nearest unmeasured candidate* (L2 distance) as the proposal.
       This maps Ax's continuous suggestion back onto the discrete pool.

    Inspired by Honegumi (https://honegumi.readthedocs.io), which uses the same
    Ax / BoTorch backend to generate BO scripts for experimental sciences.
    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals,
                 minimization, output_res):
        self.input_file    = input_file
        self.output_file   = output_file
        self.num_objectives = num_objectives
        self.num_proposals  = num_proposals
        self.minimization   = minimization if minimization is not None else False
        self.output_res     = output_res

    # ── data loading (same pattern as all other ai_tool_* modules) ────────────

    def load_data(self):
        arr = np.genfromtxt(self.input_file, skip_header=1, delimiter=",")

        arr_train = arr[~np.isnan(arr[:, -1]), :]
        arr_test  = arr[ np.isnan(arr[:, -1]), :]

        X_train   = arr_train[:, : -self.num_objectives]
        t_train   = arr_train[:,   -self.num_objectives:]
        X_all     = arr[:,          : -self.num_objectives]

        test_actions  = np.where(np.isnan(arr[:, -1]))[0].tolist()
        all_actions   = list(range(len(X_all)))
        train_actions = np.sort(list(set(all_actions) - set(test_actions)))

        return t_train, X_all, train_actions, test_actions

    # ── BO via Ax ─────────────────────────────────────────────────────────────

    def calc_ai(self, t_train, X_all, train_actions, test_actions):
        n_features = X_all.shape[1]

        # Normalise feature space to [0, 1]
        scaler     = MinMaxScaler()
        scaler.fit(X_all)
        X_norm     = scaler.transform(X_all)

        # Build Ax experiment
        parameters = [
            {"name": f"x{i}", "type": "range",
             "bounds": [0.0, 1.0], "value_type": "float"}
            for i in range(n_features)
        ]

        client = AxClient(verbose_logging=False)
        client.create_experiment(
            name="nimo_discrete_bo",
            parameters=parameters,
            objectives={"y": ObjectiveProperties(minimize=self.minimization)},
        )

        # Register training observations
        for action, t_vec in zip(train_actions, t_train):
            params = {f"x{j}": float(X_norm[action, j]) for j in range(n_features)}
            _, trial_idx = client.attach_trial(parameters=params)
            # Use first objective (single-objective mode)
            y_val = float(t_vec[0]) if self.minimization else -float(t_vec[0])
            client.complete_trial(
                trial_index=trial_idx,
                raw_data={"y": (y_val, None)},
            )

        # Ask Ax for the next suggestion
        actions = []
        remaining_test = list(test_actions)

        for _ in range(self.num_proposals):
            if not remaining_test:
                break

            suggested, _ = client.get_next_trial()
            x_sug = np.array([suggested[f"x{j}"] for j in range(n_features)])

            # Map suggestion → nearest unmeasured candidate
            X_remaining = X_norm[remaining_test]
            dists       = np.linalg.norm(X_remaining - x_sug, axis=1)
            best_local  = int(np.argmin(dists))
            best_action = remaining_test[best_local]
            actions.append(best_action)

            # Mark as "observed" so next proposal avoids it
            params = {f"x{j}": float(X_norm[best_action, j]) for j in range(n_features)}
            _, trial_idx = client.attach_trial(parameters=params)
            # Placeholder value (we don't know the real result yet)
            client.complete_trial(
                trial_index=trial_idx,
                raw_data={"y": (float(np.mean([t[0] for t in t_train])), None)},
            )
            remaining_test.pop(best_local)

        # Optional: write acquisition scores to output_res.csv
        if self.output_res:
            X_test_orig = X_all[test_actions]
            X_test_norm = X_norm[test_actions]
            scores = []
            for j in range(len(test_actions)):
                x_j   = np.array([X_test_norm[j, k] for k in range(n_features)])
                score = -float(np.linalg.norm(
                    x_j - np.array([suggested[f"x{k}"] for k in range(n_features)])
                ))
                scores.append(score)

            with open(self.input_file) as f:
                header = next(csv.reader(f))
            header = header[: -self.num_objectives] + ["ax_score"]

            rows = [[*X_test_orig[j].tolist(), scores[j]]
                    for j in range(len(test_actions))]

            with open("output_res.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)

        return actions

    # ── public entry point (same interface as every other ai_tool_*) ──────────

    def select(self):
        print("Start selection of proposals by AX (Ax/BoTorch GP)!")

        t_train, X_all, train_actions, test_actions = self.load_data()

        actions = self.calc_ai(
            t_train=t_train, X_all=X_all,
            train_actions=train_actions, test_actions=test_actions,
        )

        proposals_all = []
        with open(self.input_file) as f:
            indexes = f.readlines()[0].rstrip("\n").split(",")
        indexes = ["actions"] + indexes[: -self.num_objectives]
        proposals_all.append(indexes)

        print("Proposals")
        for i, action in enumerate(actions):
            row = [str(X_all[action][j]) for j in range(len(X_all[action]))]
            row = [str(action)] + row
            proposals_all.append(row)
            print("###")
            print("number =", i + 1)
            print("actions = ", action)
            print("proposal = ", X_all[action])
            print("###")

        with open(self.output_file, "w", newline="") as f:
            csv.writer(f).writerows(proposals_all)

        print("Finish selection of proposals by AX!")
        return "True"
