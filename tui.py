"""Full-screen terminal interface for the trained character models."""

from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Frame, Label, TextArea

from model import CharRNN
from lstm_model import CharLSTM


RNN_CHECKPOINT = Path(__file__).with_name("alice_char_rnn.npz")
LSTM_CHECKPOINT = Path(__file__).with_name("alice_char_lstm.npz")


def build_app(model, input=None, output=None, alternate_model=None):
    prompt = TextArea(text="Alice was", height=1, multiline=False, prompt=" > ")
    temperature = TextArea(text="0.6", height=1, multiline=False)
    length = TextArea(text="300", height=1, multiline=False)
    output_field = TextArea(text="Press Enter or F5 to generate from the prompt.", read_only=True,
                            scrollbar=True, wrap_lines=True, focusable=True)
    status = Label(text="Tab: fields  |  F2: temperature  |  F3: length  |  F5: generate  |  F7: switch model")
    generation = 0
    current_model = model

    def model_name():
        return "LSTM" if isinstance(current_model, CharLSTM) else "RNN"

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
            output_field.text = current_model.generate(entered, n, temp, seed=generation)
            status.text = (f"Generated {n} characters  |  sample #{generation}  |  "
                           f"{model_name()}: character-level model, expect some errors")
        except ValueError as exc:
            status.text = f"Input error: {exc}"

    def clear():
        output_field.text = ""
        status.text = "Output cleared."

    def switch_model():
        nonlocal current_model
        if alternate_model is None:
            status.text = "Only one model checkpoint is available."
            return
        current_model = alternate_model if current_model is model else model
        status.text = f"Switched to {model_name()} ({current_model.parameter_count:,} parameters). Press F5 to generate."

    prompt.buffer.accept_handler = lambda _: (generate(), True)[1]
    kb = KeyBindings()

    @kb.add("f5")
    def _generate(event):
        generate()

    @kb.add("f6")
    def _clear(event):
        clear()

    @kb.add("f7")
    def _switch_model(event):
        switch_model()

    @kb.add("c-q")
    def _quit(event):
        event.app.exit()

    fields = (prompt, temperature, length)

    def move_field(event, direction):
        layout = event.app.layout
        current = next((i for i, field in enumerate(fields) if layout.has_focus(field)), -1)
        layout.focus(fields[(current + direction) % len(fields)])

    @kb.add("tab")
    def _next_field(event):
        move_field(event, 1)

    @kb.add("s-tab")
    def _previous_field(event):
        move_field(event, -1)

    @kb.add("f2")
    def _focus_temperature(event):
        event.app.layout.focus(temperature)

    @kb.add("f3")
    def _focus_length(event):
        event.app.layout.focus(length)

    @kb.add("f4")
    def _focus_prompt(event):
        event.app.layout.focus(prompt)

    buttons = VSplit([
        Button(text="Generate [F5]", handler=generate),
        Button(text="Clear [F6]", handler=clear),
        Button(text="Switch [F7]", handler=switch_model),
        Button(text="Quit [Ctrl-Q]", handler=lambda: app.exit()),
    ], padding=2)
    body = HSplit([
        Window(FormattedTextControl(lambda: [
            ("class:title", f"  ALICE  /  {model_name()} LANGUAGE MODEL"),
            ("class:subtitle", f"     {current_model.parameter_count:,} parameters · next-character prediction")
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
    if LSTM_CHECKPOINT.exists():
        model = CharLSTM.load(LSTM_CHECKPOINT)
        alternate = CharRNN.load(RNN_CHECKPOINT) if RNN_CHECKPOINT.exists() else None
    elif RNN_CHECKPOINT.exists():
        model = CharRNN.load(RNN_CHECKPOINT)
        alternate = None
    else:
        raise SystemExit("Checkpoint missing. Run train.py or train_lstm.py first.")
    build_app(model, alternate_model=alternate).run()


if __name__ == "__main__":
    main()
