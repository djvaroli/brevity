from typing import Optional
from rich.console import Console
from rich.markdown import Markdown



def print_to_console(
    content: str,
    color: str = "green",
    accent: str = "bold",
    heading: Optional[str] = None
) -> None:
    console = Console()
    if heading:
        console.print(
            Markdown(f"# {heading}"), 
            style=f"{color} {accent}"
        )
    
    console.print(
        Markdown(content), 
        style=f"{color} {accent}"
    )
