import numpy as np
import pandas as pd

def make_val_windows(df,
                 features,
                 target_features,
                 input_steps = 30,
                 output_steps = 10
                 ):
    X_list = []
    y_list = []
    for traj_id, traj_df in df.groupby("trajectory_ID", sort=False):
        traj_df = traj_df.sort_values("TIMESTAMP")
        X_values = traj_df[features].to_numpy(dtype=np.float32)
        y_values = traj_df[target_features].to_numpy(dtype=np.float32)
        n = len(traj_df)
        section = input_steps + output_steps
        for i in range(section, n+1, section):
            # Past 30 minutes: 5 features per time step
            input_window = X_values[
                i - section : i - output_steps
            ]

            # Next 10 minutes: predict X and Y
            output_window = y_values[
                i - output_steps :
                i
            ]

            X_list.append(input_window)
            y_list.append(output_window)

        if not X_list:
            raise ValueError(
                "No complete windows found. Check trajectory lengths."
            )

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)

    return X, y

def val_df2windows(df_X,
                   df_y,
                   features,
                   target_features, cluster_number) :
    X_list = []
    y_list = []
    traj_cluster_X = df_X[df_X["assigned_cluster"] == cluster_number]
    traj_cluster_y = df_y[df_y["index"].isin(traj_cluster_X["index"].unique().tolist())]
    for idx in traj_cluster_X["index"].unique():
        section_X = traj_cluster_X[traj_cluster_X["index"] == idx][features].to_numpy(dtype=np.float32)
        section_y = traj_cluster_y[traj_cluster_y["index"] == idx][target_features].to_numpy(dtype=np.float32)
        X_list.append(section_X)
        y_list.append(section_y)


    return np.stack(X_list).astype(np.float32), np.stack(y_list).astype(np.float32)




def make_windows(df,
                 features,
                 target_features,
                 input_steps = 30,
                 output_steps = 10
                 ):
    X_list = []
    y_list = []
    for traj_id, traj_df in df.groupby("trajectory_ID", sort=False):
        traj_df = traj_df.sort_values("TIMESTAMP")
        X_values = traj_df[features].to_numpy(dtype=np.float32)
        y_values = traj_df[target_features].to_numpy(dtype=np.float32)
        n = len(traj_df)
        for i in range(n - input_steps-output_steps+1):
            # Past 30 minutes: 5 features per time step
            input_window = X_values[
                i : i + input_steps
            ]

            # Next 10 minutes: predict X and Y
            output_window = y_values[
                i + input_steps :
                i + input_steps + output_steps
            ]

            X_list.append(input_window)
            y_list.append(output_window)

    if not X_list:
        raise ValueError(
            "No complete windows found. Check trajectory lengths."
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)

    return X, y