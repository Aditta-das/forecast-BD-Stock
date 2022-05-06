def sliding_window(ts, features):
    X = []
    Y = []

    for i in range(features + 1, len(ts)+1):
        X.append(ts[i - (features + 1) : i - 1])
        Y.append([ts[i - 1]])
    return X, Y

