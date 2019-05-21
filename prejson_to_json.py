"""
Passage des fichiers prejson vers json

@date: 21/05/2019
@author: Quentin Lieumont
"""


def build(path: str, path_to: str = ""):
    """
    :type path: path to the .prejson file
    :param path_to: optional output file
    :return: None
    """
    if path_to is "":
        if path.endswith(".prejson"):
            path_to = path[:-8]
        else:
            path_to = path
        path_to += ".json"
    with open(path_to, "w") as w:
        w.write("[\n")
    with open(path, "r") as r:
        for line in r:
            header, end = line.split("[[")
            end = "[[" + end
            end = "".join(end.split("[nan, nan, nan, nan], "))
            end = "".join(end.split(", [nan, nan, nan, nan]"))
            with open(path_to, "a") as w:
                w.write(header + end)
    with open(path_to, "a") as w:
        w.write("{}]\n")


if __name__ == "__main__":
    import os

    PATH = "data/json/"
    os.chdir(PATH)
    listed = [e for e in os.listdir("./") if e.endswith(".prejson")]

    for l in listed:
        build(l)
