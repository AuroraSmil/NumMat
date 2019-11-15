import pickle

import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open("numerical", "rb") as f:
        data = pickle.load(f)
        all_numerical, all_exact, all_diff = data

    errs = {}
    max_errs = {}

    for i, (meta, data) in enumerate(all_diff):
        theta, n = meta["theta"], meta["n"]
        max_diffs = []
        for frame in data:
            max_diffs.append(frame.max())
            if theta not in errs:
                errs[theta] = [(n, frame.max())]
            else:
                errs[theta].append((n, frame.max()))

        if theta not in max_errs:
            max_errs[theta] = [(n, max(max_diffs))]
        else:
            max_errs[theta].append((n, max(max_diffs)))

    for theta, err in errs.items():
        fig, axes = plt.subplots()
        axes.plot(*list(zip(*err)), "--")
        axes.scatter(*list(zip(*max_errs[theta])), c="red", label="max error")
        axes.scatter(*list(zip(*err)), label="error pr frame", s=7)
        axes.set_title("theta: " + str(theta))
        axes.legend()

        print(theta)

    plt.show()
