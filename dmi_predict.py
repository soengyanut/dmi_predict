from rich.console import Group
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Header,
    Input,
    OptionList,
    Static,
)
from textual.widgets.option_list import Option
from src.model_loader import load_model
from src.prediction import predict_disease


BLUE = "#1D4ED8"
DARK_BLUE = "#17365D"

EMPTY_MSG = "Select your symptoms to begin."

class DMIPredict(App):
    CSS_PATH = "dmi_predict.tcss"
    TITLE = "DMI-Predict"
    SUB_TITLE = "AI Disease Prediction System"

    BINDINGS = [
        ("a", "add", "Add"),
        ("r", "remove", "Remove"),
        ("c", "clear_all", "Clear"),
        ("p", "predict", "Predict"),
        ("/", "focus_search", "Search"),
        ("escape", "focus_available", "Available"),
        ("ctrl+tab", "switch_panel", "Switch panel"),
        ("q", "quit", "Exit"),
    ]

    def __init__(self):
        super().__init__()

        self.system = None
        self.features = []
        self.selected = set()

    # ==================================================
    # USER INTERFACE
    # ==================================================

    def compose(self) -> ComposeResult:

        yield Header()

        with Vertical(id="root"):

            yield Static(
                "DMI-PREDICT | AI DISEASE PREDICTION SYSTEM",
                id="banner"
            )

            with Horizontal(id="panels"):

                with Vertical(
                    id="available-panel",
                    classes="panel"
                ):

                    yield Input(
                        placeholder="Search all symptoms...",
                        id="search"
                    )

                    yield OptionList(id="available")

                with Vertical(
                    id="selected-panel",
                    classes="panel"
                ):

                    yield OptionList(id="selected")

            with Horizontal(id="actions"):

                yield Button(
                    "ADD +",
                    id="add",
                    classes="primary"
                )

                yield Button("REMOVE -", id="remove", classes="primary")
                yield Button("CLEAR ALL", id="clear", classes="primary")
                yield Button("PREDICT", id="predict", classes="primary")
                yield Button("EXIT", id="exit", classes="primary")

            yield Static(
                EMPTY_MSG,
                id="result"
            )

            yield Static(
                "Loading model...",
                id="hint"
            )

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def on_mount(self):

        self.theme = "textual-light"

        self.query_one(
            "#available-panel"
        ).border_title = "Available Symptoms"

        self.query_one(
            "#selected-panel"
        ).border_title = "Your Selected Symptoms"

        self.query_one(
            "#result"
        ).border_title = "Prediction Result"

        try:

            self.system = load_model()

            if not self.system:
                raise ValueError("Model package is empty.")

            self.features = list(
                self.system["feature_columns"]
            )

            if not self.features:
                raise ValueError("No symptoms were loaded.")

            self.features.sort(
                key=lambda s: self.display_name(s).casefold()
            )

            self.refresh_lists()

            self.set_hint(
                f"Model loaded | {len(self.features)} symptoms available"
            )

            self.query_one(
                "#available",
                OptionList
            ).focus()

        except Exception as error:

            self.show_error(
                f"Model loading failed: "
                f"{type(error).__name__}: {error}"
            )

    # ==================================================
    # DISPLAY HELPERS
    # ==================================================

    @staticmethod
    def display_name(symptom):

        return str(symptom).replace("_", " ").title()

    def set_hint(self, message):

        self.query_one(
            "#hint",
            Static
        ).update(message)

    def reset_result(self):

        result = self.query_one(
            "#result",
            Static
        )

        result.remove_class("has-result")
        result.update(EMPTY_MSG)

    def show_error(self, message):

        result = self.query_one(
            "#result",
            Static
        )

        result.remove_class("has-result")

        result.update(
            Group(
                Text(
                    "PREDICTION ERROR",
                    style="bold #B42318"
                ),
                Text(
                    str(message),
                    style=DARK_BLUE
                )
            )
        )

        self.set_hint("Please check the error message.")

    # ==================================================
    # SYMPTOM LISTS
    # ==================================================

    def refresh_lists(self):

        available = self.query_one(
            "#available",
            OptionList
        )

        selected = self.query_one(
            "#selected",
            OptionList
        )

        query = self.query_one(
            "#search",
            Input
        ).value.strip().casefold()

        old_available = available.highlighted
        old_selected = selected.highlighted

        available.clear_options()
        selected.clear_options()

        visible = [
            symptom
            for symptom in self.features
            if query in self.display_name(symptom).casefold()
        ]

        selected_items = sorted(
            self.selected,
            key=lambda s: self.display_name(s).casefold()
        )

        available.add_options(
            Option(
                Text(
                    f"{'[✓]' if symptom in self.selected else '[ ]'}  "
                    f"{self.display_name(symptom)}",
                    style=(
                        "bold #1D4ED8"
                        if symptom in self.selected
                        else DARK_BLUE
                    )
                ),
                id=symptom
            )
            for symptom in visible
        )

        selected.add_options(
            Option(
                Text(
                    f"[✓]  {self.display_name(symptom)}",
                    style="bold #1D4ED8"
                ),
                id=symptom
            )
            for symptom in selected_items
        )

        if available.option_count:
            available.highlighted = min(
                old_available if old_available is not None else 0,
                available.option_count - 1
            )

        if selected.option_count:
            selected.highlighted = min(
                old_selected if old_selected is not None else 0,
                selected.option_count - 1
            )

        self.query_one(
            "#available-panel"
        ).border_subtitle = f"{len(visible)} available"

        self.query_one(
            "#selected-panel"
        ).border_subtitle = f"{len(selected_items)} selected"

    # ==================================================
    # SEARCH
    # ==================================================

    def on_input_changed(self, event: Input.Changed):

        if event.input.id == "search":
            self.refresh_lists()

    # ==================================================
    # KEYBOARD NAVIGATION
    # ==================================================

    def on_key(self, event):

        if (
            event.key == "down"
            and isinstance(self.focused, Input)
            and self.focused.id == "search"
        ):

            event.prevent_default()
            event.stop()

            self.query_one(
                "#available",
                OptionList
            ).focus()

    def action_focus_search(self):

        self.query_one("#search", Input).focus()

    def action_focus_available(self):

        self.query_one("#available", OptionList).focus()

    def action_switch_panel(self):

        available = self.query_one(
            "#available",
            OptionList
        )

        selected = self.query_one(
            "#selected",
            OptionList
        )

        if self.focused is available:
            selected.focus()
        else:
            available.focus()

    # ==================================================
    # ADD SYMPTOM
    # ==================================================

    def action_add(self):

        widget = self.query_one(
            "#available",
            OptionList
        )

        index = widget.highlighted

        if index is None:
            self.set_hint("Highlight an available symptom first.")
            return

        symptom = widget.get_option_at_index(index).id

        self.selected.add(symptom)

        self.reset_result()
        self.refresh_lists()

        self.query_one(
            "#available",
            OptionList
        ).focus()

        self.set_hint(
            f"Added: {self.display_name(symptom)}"
        )

    # ==================================================
    # REMOVE SYMPTOM
    # ==================================================

    def action_remove(self):

        widget = self.query_one(
            "#selected",
            OptionList
        )

        index = widget.highlighted

        if index is None:
            self.set_hint("Highlight a selected symptom first.")
            return

        symptom = widget.get_option_at_index(index).id

        self.selected.discard(symptom)

        self.reset_result()
        self.refresh_lists()

        self.query_one(
            "#selected",
            OptionList
        ).focus()

        self.set_hint(
            f"Removed: {self.display_name(symptom)}"
        )

    # ==================================================
    # CLEAR ALL
    # ==================================================

    def action_clear_all(self):

        self.selected.clear()

        self.reset_result()
        self.refresh_lists()

        self.set_hint("All selected symptoms cleared.")

    # ==================================================
    # MACHINE LEARNING PREDICTION
    # ==================================================

    def action_predict(self):

        try:
            self.set_hint("Starting disease prediction...")

            if self.system is None:
                raise ValueError("SVM model is not loaded.")

            if len(self.selected) == 0:
                raise ValueError("Please select symptoms first.")

            # Send selected symptoms to the trained model.
            disease, _ = predict_disease(
                self.system,
                self.selected
            )

            # Display result.
            result = self.query_one("#result", Static)

            result.update(
                Group(
                    Text("PREDICTED DISEASE", style="bold #42658F"),
                    Text(""),
                    Text(str(disease).upper(), style="bold #174EA6"),
                )
            )

            result.add_class("has-result")

            self.set_hint("Prediction completed successfully.")

        except Exception as error:

            self.show_error(
                f"{type(error).__name__}: {error}"
            )

    def on_button_pressed(self, event: Button.Pressed):

        actions = {
            "add": self.action_add,
            "remove": self.action_remove,
            "clear": self.action_clear_all,
            "predict": self.action_predict,
            "exit": self.exit,
        }
        action = actions.get(event.button.id)
        if action is not None:
            action()

    # ==================================================
    # ENTER TO ADD / REMOVE
    # ==================================================

    def on_option_list_option_selected(
        self,
        event: OptionList.OptionSelected
    ):

        symptom = event.option.id

        if not symptom:
            return

        if event.option_list.id == "available":

            if symptom in self.selected:
                self.selected.remove(symptom)
                status = "Deselected"
            else:
                self.selected.add(symptom)
                status = "Selected"

            self.set_hint(
                f"{status}: {self.display_name(symptom)}"
            )

        elif event.option_list.id == "selected":

            self.selected.discard(symptom)

            self.set_hint(
                f"Removed: {self.display_name(symptom)}"
            )

        self.reset_result()
        self.refresh_lists()


if __name__ == "__main__":
    DMIPredict().run()
