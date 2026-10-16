"""Full-screen terminal interface for the trained character RNN."""

from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Frame, Label, TextArea

from model import CharRNN


CHECKPOINT = Path(__file__).with_name("alice_char_rnn.npz")


def build_app(model, input=None, output=None):
    prompt = TextArea(text="Alice was", height=1, multiline=False, prompt=" > ")
    temperature = TextArea(text="0.6", height=1, multiline=False)
    length = TextArea(text="300", height=1, multiline=False)
    output_field = TextArea(text="Press Enter or F5 to generate from the prompt.", read_only=True,
                            scrollbar=True, wrap_lines=True, focusable=True)
    status = Label(text="Ready  |  trained from scratch  |  Ctrl-Q: quit")
    generation = 0

    def generate():
        nonlocal generation
        entered = prompt.text.strip()
        try:
            n = int(length.text.strip())
            temp = float(temperature.text.strip())
            if not entered:
                raise ValueError("Enter a prompt first.")
            if not 1 <= n <= 2000:
                raise ValueError("Length must be from 1 to 2000.")
            if not 0.1 <= temp <= 2.0:
                raise ValueError("Temperature must be from 0.1 to 2.0.")
            generation += 1
            output_field.text = model.generate(entered, n, temp, seed=generation)
            status.text = f"Generated {n} characters  |  sample #{generation}  |  F5: generate again"
        except ValueError as exc:
            status.text = f"Input error: {exc}"

    def clear():
        output_field.text = ""
        status.text = "Output cleared."

    prompt.buffer.accept_handler = lambda _: (generate(), True)[1]
    kb = KeyBindings()

    @kb.add("f5")
    def _generate(event):
        generate()

    @kb.add("f6")
    def _clear(event):
        clear()

    @kb.add("c-q")
    def _quit(event):
        event.app.exit()

    buttons = VSplit([
        Button(text="Generate [F5]", handler=generate),
        Button(text="Clear [F6]", handler=clear),
        Button(text="Quit [Ctrl-Q]", handler=lambda: app.exit()),
    ], padding=2)
    body = HSplit([
        Window(FormattedTextControl([
            ("class:title", "  ALICE  /  SMALL LANGUAGE MODEL"),
            ("class:subtitle", "     24,505 parameters · next-character prediction")
        ]), height=2),
        Frame(prompt, title="PROMPT"),
        VSplit([
            Frame(temperature, title="TEMPERATURE  (0.1–2.0)"),
            Frame(length, title="NEW CHARACTERS  (1–2000)"),
        ], padding=1, height=3),
        buttons,
        Frame(output_field, title="GENERATED TEXT"),
        Window(FormattedTextControl(lambda: status.text), height=1, style="class:status"),
    ])
    app = Application(
        layout=Layout(body, focused_element=prompt),
        key_bindings=kb,
        style=Style.from_dict({
            "": "bg:#101827 #e8edf3",
            "frame.border": "#608cc5",
            "frame.label": "bold #78aee8",
            "title": "bold #ffffff",
            "subtitle": "#a2b8d2",
            "button": "bg:#254c73 #ffffff",
            "button.focused": "bg:#73b3ec #0d1825",
            "status": "bg:#213b55 #ffffff",
        }),
        full_screen=True,
        mouse_support=True,
        input=input,
        output=output,
    )
    return app


def main():
    if not CHECKPOINT.exists():
        raise SystemExit("Checkpoint missing. Run train.py first.")
    model = CharRNN.load(CHECKPOINT)
    build_app(model).run()


if __name__ == "__main__":
    main()
