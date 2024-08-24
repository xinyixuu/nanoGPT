with open("tokens.txt", "w") as file:
    for i in range(1, 4096):
        file.write(f"<snac>{i}</snac>\n")