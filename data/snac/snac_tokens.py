with open("tokens.txt", "w") as file:
    for i in range(0, 4098):
        file.write(f"<snac>{i}</snac>\n")
