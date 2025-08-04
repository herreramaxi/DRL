def printc(text: str, color: str = "green", bold: bool = False, end = "\n"):   
    color_dict = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37
    }
    base = color_dict.get(color.lower(), 37)
    style = 1 if bold else 0
    prefix = f"\033[{style};{base}m"
    suffix = "\033[0m"
    print(f"{prefix}{text}{suffix}", end=end)

def info(text: str):   
    printc(text, "white")

def important(text: str):   
    printc("-> " + text, "cyan")

def important2(text: str):   
    printc("-> " + text, "magenta")

def success(text: str):   
    printc("-> " + text, "green")

def error(text: str):   
    printc(text, "red")

# Example usage:
if __name__ == "__main__":
    # printc("✔ Success!",   color="green")
    # printc("✖ Failure...", color="red", bold=True)
    # printc("Note in cyan", color="cyan")
    # print("hey")
    # printc("✔ Success!", "green")
    # printc("✔ Success!", "green", True)
    print("This is a normal print message")
    info("This is an info message")
    important("This is a important message")
    important2("This is a important2 message")
    success("This is a success message")
    error("This is a error message")