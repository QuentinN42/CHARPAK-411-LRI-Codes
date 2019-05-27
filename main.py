from Obj2.real import build


if __name__ == "__main__":
    net = build(
        save_folder="data/Obj2/real/remove1",
        n_epochs=100,
        n_build=25
    )
