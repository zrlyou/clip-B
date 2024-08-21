from rich import print


def red_print(text):
    print(f"[red]{text}[/red]")
    
def green_print(text):
    print(f"[green]{text}[/green]")
    
def yellow_print(text):
    print(f"[yellow]{text}[/yellow]")

def blue_print(text):
    print(f"[blue]{text}[/blue]")

if __name__ == "__main__":
    yellow_print("This is red text.")