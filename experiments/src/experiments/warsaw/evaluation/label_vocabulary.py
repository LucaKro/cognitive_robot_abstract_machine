"""
Reconciling our ontology's class names with somebody else's label vocabulary.

A dataset's labels and our classes were written by different people for different
purposes: HM3D says ``kitchen cabinet`` where the ontology says ``Cabinet``. Counting
the two against each other without reconciling them first counts every difference of
wording as a mistake, which measures the annotator's phrasing rather than the run.

What is reconciled here is wording and nothing else. Where the ontology carves the world
more coarsely than the dataset does -- answering ``Decor`` for a light fixture -- the
two are left apart, because that is a disagreement about the world rather than about
words and it is the thing an evaluation exists to show.

Two things can decide it. :class:`LexicalMatcher` reads the words, which settles
``kitchen cabinet`` against ``Cabinet`` and nothing a dictionary would be needed for.
:class:`EmbeddingMatcher` reads what the names mean, which is what says a fridge is a
refrigerator, and falls back on the wording where the likeness is only middling -- the
rule the earlier HM3D study used, whose paper reports a threshold its code does not use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from typing_extensions import Iterable, List, Optional, Protocol, Sequence, Set

WORDS = re.compile(r"[A-Z](?:[a-z0-9]*)|[a-z0-9]+")
"""
What a name is made of, whether it is camel-cased or spaced.
"""


def spoken_class_name(class_name: str) -> str:
    """
    Say a class of the ontology the way a dataset would write it.

    :param class_name: The class, as the ontology spells it.
    :return: Its words, lowercased and spaced.
    """
    return " ".join(WORDS.findall(class_name or "")).lower()


# %% what any of them has to do


class Encoder(Protocol):
    """
    Something that turns names into the places they mean.
    """

    def __call__(self, names: Sequence[str]) -> np.ndarray:
        """
        :param names: The names to place.
        :return: Their places, as unit vectors, one row apiece.
        """


class Matcher(Protocol):
    """
    Something that finds what a name is called in another vocabulary.
    """

    def matched(self, name: str, vocabulary: Iterable[str]) -> Optional[str]:
        """
        :param name: The name to place.
        :param vocabulary: The names it may be one of.
        :return: The one it is, or None when it is none of them.
        """


# %% when two names mean the same kind of thing


@dataclass
class LexicalMatcher:
    """
    Whether two names mean the same kind of thing, judged by their words alone.
    """

    agreement: float = 0.5
    """
    What share of the two names' words must be shared for them to be one thing.

    Half, so that ``kitchen cabinet`` reaches ``cabinet`` while ``coffee table`` does
    not reach ``coffee machine``. Sharing one word of two is the commonest way a purely
    lexical match goes wrong.
    """

    language: str = "english"
    """
    The language whose stems fold a word onto its inflections.
    """

    shared_beginning: float = 0.8
    """
    What share of the longer word two words must begin alike in to be one word.

    Stemming folds most inflections together, but not all of them: it leaves
    ``shelving`` as ``shelv`` beside ``shelf``, which agree in four letters of five.
    """

    @cached_property
    def stemmer(self) -> SnowballStemmer:
        """
        :return: What folds ``shelving`` onto ``shelf``.
        """
        return SnowballStemmer(self.language)

    def stems(self, name: str) -> Set[str]:
        """
        :param name: A class name or a label.
        :return: Its words, stemmed, each once.
        """
        return {self.stemmer.stem(word) for word in WORDS.findall(name or "")}

    def agree(self, one: str, other: str) -> float:
        """
        Measure how far two names say the same thing.

        :param one: A name.
        :param other: Another.
        :return: The share of their words the two have in common, one when a single word
            is a prefix of the other long enough to be the same word differently ended.
        """
        here, there = self.stems(one), self.stems(other)
        if not here or not there:
            return 0.0
        if here == there:
            return 1.0
        shared = len(here & there) / len(here | there)
        if len(here) == 1 and len(there) == 1:
            return max(shared, self.same_word_differently_ended(here, there))
        return shared

    def same_word_differently_ended(self, one: Set[str], other: Set[str]) -> float:
        """
        :param one: One stemmed word.
        :param other: Another.
        :return: One where the two begin alike enough to be the same word, else zero.
        """
        here, there = next(iter(one)), next(iter(other))
        alike = 0
        for mine, yours in zip(here, there):
            if mine != yours:
                break
            alike += 1
        return (
            1.0 if alike / max(len(here), len(there)) >= self.shared_beginning else 0.0
        )

    def matched(self, name: str, vocabulary: Iterable[str]) -> Optional[str]:
        """
        Find what a name is called in another vocabulary.

        :param name: The name to place.
        :param vocabulary: The names it may be one of.
        :return: The closest of them that agrees enough, or None when none does.
        """
        scored = [(self.agree(name, one), one) for one in vocabulary]
        best = max(scored, default=(0.0, ""), key=lambda one: (one[0], -len(one[1])))
        return best[1] if best[0] >= self.agreement else None


# %% when two names mean the same thing


@dataclass
class EmbeddingMatcher:
    """
    Whether two names mean the same kind of thing, judged by what they mean.

    A fridge is a refrigerator and no reading of the letters says so, which is what this
    is for. Below the point where likeness alone settles it, the wording has to agree as
    well, so that a merely related name is not read as the same one.
    """

    settles_it: float = 0.65
    """
    How alike two names must mean for that alone to make them one thing.

    The earlier HM3D study settled a pair outright at 0.75 and this is looser, because
    that left ``lamp`` and ``light fixture`` apart at 0.669 and they are one thing --
    seventeen objects across three rooms, every one of them a correct answer counted
    wrong. The rule is otherwise theirs.
    """

    worth_considering: float = 0.55
    """
    How alike they must mean to be one thing if their wording agrees too.

    Theirs, unchanged. It carries a pair the meaning alone would not settle, which is
    what tells ``shelving unit`` from ``shelf`` and keeps ``kitchen counter`` from
    ``worktop``, alike in meaning and sharing no word.
    """

    wording: LexicalMatcher = field(default_factory=LexicalMatcher)
    """
    What is asked whether the wording agrees, between the two.
    """

    model_name: str = "all-MiniLM-L6-v2"
    """
    The encoder to read meanings with, when none is given.
    """

    encoder: Optional[Encoder] = None
    """
    What turns names into the places they mean, given rather than downloaded where a
    caller has one.
    """

    @cached_property
    def encode(self) -> Encoder:
        """
        :return: What turns names into the places they mean. Downloading the encoder is
            put off until something is actually asked, so holding one of these costs
            nothing.
        """
        if self.encoder is not None:
            return self.encoder
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name)
        return lambda names: model.encode(list(names), normalize_embeddings=True)

    def likeness(self, name: str, vocabulary: Sequence[str]) -> np.ndarray:
        """
        :param name: The name to place.
        :param vocabulary: The names to place it among.
        :return: How alike it means to each of them, from minus one to one.
        """
        placed = self.encode([name, *vocabulary])
        return placed[1:] @ placed[0]

    def matched(self, name: str, vocabulary: Iterable[str]) -> Optional[str]:
        """
        :param name: The name to place.
        :param vocabulary: The names it may be one of.
        :return: The one it means most nearly, where that is near enough, else None.
        """
        vocabulary = list(vocabulary)
        if not vocabulary:
            return None
        alike = self.likeness(name, vocabulary)
        best = int(np.argmax(alike))
        candidate, closeness = vocabulary[best], float(alike[best])
        if closeness >= self.settles_it:
            return candidate
        if closeness >= self.worth_considering and self.wording.agree(name, candidate):
            return candidate
        return None


# %% putting one bag of labels into the other's words


def to_vocabulary(
    names: Sequence[str],
    vocabulary: Iterable[str],
    matcher: Optional[Matcher] = None,
) -> List[str]:
    """
    Say a bag of names in another vocabulary's words, keeping what it has no word for.

    An unmatched name is kept as it was rather than dropped: it is still something the
    run answered, and dropping it would quietly raise the score.

    :param names: The names to move.
    :param vocabulary: The words to move them into.
    :param matcher: What decides that two names are one thing, defaulting to the
        wording alone.
    :return: Each name as the other vocabulary says it, or unchanged.
    """
    matcher = matcher if matcher is not None else LexicalMatcher()
    vocabulary = list(vocabulary)
    return [matcher.matched(one, vocabulary) or one for one in names]
