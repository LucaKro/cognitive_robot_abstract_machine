"""
Where the pipeline's generated text is written from.

Two kinds of thing are generated here: Python a run leaves behind, and documents -- a
report, and the message each question puts to a model. Both were assembled in Python
before, which reads badly for the same reason a prompt written into the middle of a
module does: the shape of the document is buried in the code that emits it, and nobody
reads it before changing the code around it.

The templates are Jinja, rendered through krrood's generator, which is what the rest of
the repository already generates code with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from krrood.code_generation.generator import CodeGenerator
from typing_extensions import Any


@dataclass
class PipelineTemplates:
    """
    The templates a run writes its generated text from.
    """

    directory: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "templates"
    )
    """
    Where they are kept, by default beside this module.
    """

    keep_trailing_newline: bool = True
    """
    Whether a template's last newline survives rendering.

    Jinja drops it by default, which is right for a fragment pasted into something else
    and wrong for every template here: each is written out as a whole file, and a Python
    file without its final newline is one every tool reports as changed.
    """

    @property
    def generator(self) -> CodeGenerator:
        """
        :return: The renderer, reading from this directory.
        """
        made = CodeGenerator(template_directory=str(self.directory))
        made.environment.keep_trailing_newline = self.keep_trailing_newline
        return made

    def render(self, template_name: str, **context: Any) -> str:
        """
        :param template_name: The template to fill in, by file name.
        :param context: What to fill it in with.
        :return: The text it makes.
        """
        return self.generator.render(template_name, **context)

    def render_document(self, template_name: str, **context: Any) -> str:
        """
        Fill in a template whose text is put into a message rather than written to a
        file.

        The blank line a file ends with is a property of the file, not of what it says,
        and a message carrying it says the same thing with a trailing blank line. The
        prompt a model is given as its instructions is read the same way, for the same
        reason.

        :param template_name: The template to fill in, by file name.
        :param context: What to fill it in with.
        :return: The text it makes, without the newline the file ends with.
        """
        return self.render(template_name, **context).strip()
