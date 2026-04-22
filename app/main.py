"""Basic Tkinter app entrypoint."""

import tkinter as tk
from tkinter import ttk


class BasicApp(tk.Tk):
    """A minimal Tkinter window with a label and button."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ZoMBI-Hop App")
        self.geometry("420x240")
        self.minsize(360, 200)
        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=20)
        container.pack(fill="both", expand=True)

        title = ttk.Label(container, text="Tkinter app is running", font=("Segoe UI", 14))
        title.pack(pady=(0, 16))

        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(container, textvariable=self.status_var)
        status.pack(pady=(0, 16))

        button_row = ttk.Frame(container)
        button_row.pack()

        ping_btn = ttk.Button(button_row, text="Ping", command=self._on_ping)
        ping_btn.pack(side="left", padx=6)

        quit_btn = ttk.Button(button_row, text="Quit", command=self.destroy)
        quit_btn.pack(side="left", padx=6)

    def _on_ping(self) -> None:
        self.status_var.set("Hello from Tkinter.")


def main() -> None:
    app = BasicApp()
    app.mainloop()


if __name__ == "__main__":
    main()

