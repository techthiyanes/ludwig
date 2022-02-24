fraud_inputs = [
    {"name": "Time", "type": "numerical", "output_flag": False},
    {"name": "V1", "type": "numerical", "output_flag": False},
    {"name": "V2", "type": "numerical", "output_flag": False},
    {"name": "V3", "type": "numerical", "output_flag": False},
    {"name": "V4", "type": "numerical", "output_flag": False},
    {"name": "V5", "type": "numerical", "output_flag": False},
    {"name": "V6", "type": "numerical", "output_flag": False},
    {"name": "V7", "type": "numerical", "output_flag": False},
    {"name": "V8", "type": "numerical", "output_flag": False},
    {"name": "V9", "type": "numerical", "output_flag": False},
    {"name": "V10", "type": "numerical", "output_flag": False},
    {"name": "V11", "type": "numerical", "output_flag": False},
    {"name": "V12", "type": "numerical", "output_flag": False},
    {"name": "V13", "type": "numerical", "output_flag": False},
    {"name": "V14", "type": "numerical", "output_flag": False},
    {"name": "V15", "type": "numerical", "output_flag": False},
    {"name": "V16", "type": "numerical", "output_flag": False},
    {"name": "V17", "type": "numerical", "output_flag": False},
    {"name": "V18", "type": "numerical", "output_flag": False},
    {"name": "V19", "type": "numerical", "output_flag": False},
    {"name": "V20", "type": "numerical", "output_flag": False},
    {"name": "V21", "type": "numerical", "output_flag": False},
    {"name": "V22", "type": "numerical", "output_flag": False},
    {"name": "V23", "type": "numerical", "output_flag": False},
    {"name": "V24", "type": "numerical", "output_flag": False},
    {"name": "V25", "type": "numerical", "output_flag": False},
    {"name": "V26", "type": "numerical", "output_flag": False},
    {"name": "V27", "type": "numerical", "output_flag": False},
    {"name": "V28", "type": "numerical", "output_flag": False},
    {"name": "Amount", "type": "numerical", "output_flag": False},
]

fraud_outputs = [{"name": "Class", "type": "binary", "output_flag": True}]

COLS = {
    "df2_cols": {
        "V1": DFS["test_df_2"]["V1"],
        "V2": DFS["test_df_2"]["V2"],
        "V3": DFS["test_df_2"]["V3"],
        "V4": DFS["test_df_2"]["V4"],
        "V5": DFS["test_df_2"]["V5"],
        "V6": DFS["test_df_2"]["V6"],
        "V7": DFS["test_df_2"]["V7"],
        "V8": DFS["test_df_2"]["V8"],
        "V9": DFS["test_df_2"]["V9"],
        "V10": DFS["test_df_2"]["V10"],
        "V11": DFS["test_df_2"]["V11"],
        "V12": DFS["test_df_2"]["V12"],
        "V13": DFS["test_df_2"]["V13"],
        "V14": DFS["test_df_2"]["V14"],
        "V15": DFS["test_df_2"]["V15"],
        "V16": DFS["test_df_2"]["V16"],
        "V17": DFS["test_df_2"]["V17"],
        "V18": DFS["test_df_2"]["V18"],
        "V19": DFS["test_df_2"]["V19"],
        "V20": DFS["test_df_2"]["V20"],
        "V21": DFS["test_df_2"]["V21"],
        "V22": DFS["test_df_2"]["V22"],
        "V23": DFS["test_df_2"]["V23"],
        "V24": DFS["test_df_2"]["V24"],
        "V25": DFS["test_df_2"]["V25"],
        "V26": DFS["test_df_2"]["V26"],
        "V27": DFS["test_df_2"]["V27"],
        "V28": DFS["test_df_2"]["V28"],
        "Amount": DFS["test_df_2"]["Amount"],
        "Class": DFS["test_df_2"]["Class"],
        "Time": DFS["test_df_2"]["Time"],
        "split": pd.Series(np.random.choice(3, len(DFS["test_df_2"]), p=(0.7, 0.1, 0.2))),
    }
}
