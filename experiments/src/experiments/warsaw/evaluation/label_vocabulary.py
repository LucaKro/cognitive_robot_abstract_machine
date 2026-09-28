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

:class:`HeadNounMatcher` reads what the names mean as well, and refuses two names that
share a word but not the thing they name: ``wall decor`` hangs on a ``wall`` and is not
one. It is kept beside the others rather than replacing them, so a run can be scored
both ways.

The question is asked of **one pair at a time**. The study this rule comes from could
not do that: it compared two bags of label strings with no correspondence between them,
so the nearest label in the whole ground-truth vocabulary was the only target available
to it. Here each object carries HM3D's own object number, so an answer and a label are
already about the same object. Asking which label of the room an answer most resembles
instead loses a right answer to a neighbouring one -- a kitchen cabinet answered
``Cabinet`` was scored against a different object's ``cabinet`` and counted wrong,
twelve times in one room.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from itertools import takewhile

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from typing_extensions import Dict, List, Optional, Protocol, Sequence, Set, Tuple

WORDS = re.compile(r"[A-Z](?:[a-z0-9]*)|[a-z0-9]+")
"""
What a name is made of, whether it is camel-cased or spaced.
"""

PREPOSITIONS = frozenset({"of", "on", "with", "under", "in", "for", "at"})
"""
The words after which a name goes on to say something other than what the thing is: a
``tool with handle`` is a tool.
"""

CONJUNCTION = "and"
"""
The word joining two things one name names: an ``oven and stove`` is both.
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
    Something that says whether two names mean the same kind of thing.
    """

    def means_the_same(self, name: str, other: str) -> bool:
        """
        :param name: One name.
        :param other: The other.
        :return: Whether the two name the same kind of thing.
        """

    def far_apart(self, name: str, other: str) -> bool:
        """
        :param name: One name.
        :param other: The other.
        :return: Whether the two are further apart in meaning than anything the matcher
            reads the wording for, so that no rule of it could have matched them.
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

    def means_the_same(self, name: str, other: str) -> bool:
        """
        :param name: One name.
        :param other: The other.
        :return: Whether their words agree enough to be one thing.
        """
        return self.agree(name, other) >= self.agreement

    def far_apart(self, name: str, other: str) -> bool:
        """
        :param name: One name.
        :param other: The other.
        :return: False. How far apart two meanings are is not something the words say.
        """
        return False


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

    likenesses: Dict[Tuple[str, str], float] = field(
        default_factory=dict, repr=False, compare=False
    )
    """
    The likeness of every pair already asked about, since a room asks about the same
    pairs many times over.
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

    def likeness(self, name: str, other: str) -> float:
        """
        :param name: One name.
        :param other: The other.
        :return: How alike the two mean, from minus one to one.
        """
        if (name, other) not in self.likenesses:
            placed = self.encode([name, other])
            self.likenesses[(name, other)] = float(placed[1] @ placed[0])
        return self.likenesses[(name, other)]

    def means_the_same(self, name: str, other: str) -> bool:
        """
        :param name: One name.
        :param other: The other.
        :return: Whether they mean nearly enough the same to be one thing.
        """
        alike = self.likeness(name, other)
        if alike >= self.settles_it:
            return True
        return alike >= self.worth_considering and bool(self.wording.agree(name, other))

    def far_apart(self, name: str, other: str) -> bool:
        """
        :param name: One name.
        :param other: The other.
        :return: Whether the two are less alike than even a shared word would make up for.
        """
        return self.likeness(name, other) < self.worth_considering


# %% when two names sharing a word must also share what they name


@dataclass
class HeadNounMatcher:
    """
    Whether two names mean the same kind of thing, judged by what they mean and refusing
    a shared word that names different things.

    A word two names share settles nothing by itself: ``wall decor`` and ``wall`` share
    one, and so do ``door`` and ``door frame``. What a name names is its head noun, the
    last word before anything that goes on to describe it, so two names sharing a word
    have to share their head as well -- or have heads that mean the same, as a coffee
    maker is a coffee machine.
    """

    meaning: EmbeddingMatcher = field(default_factory=EmbeddingMatcher)
    """
    What judges the two names by meaning before their heads are read.
    """

    def head_nouns(self, name: str) -> List[str]:
        """
        :param name: A class name or a label.
        :return: What it names, one word per thing it joins, lowercased.
        """
        heads: List[str] = []
        for part in " ".join(WORDS.findall(name.lower())).split(f" {CONJUNCTION} "):
            described = list(
                takewhile(lambda word: word not in PREPOSITIONS, part.split())
            )
            if described:
                heads.append(described[-1])
        return heads

    def means_the_same(self, name: str, other: str) -> bool:
        """
        :param name: One name.
        :param other: The other.
        :return: Whether they mean the same, and name the same thing where they share a
            word.
        """
        if not self.meaning.means_the_same(name, other):
            return False
        wording = self.meaning.wording
        if not wording.stems(name) & wording.stems(other):
            return True
        heads, other_heads = self.head_nouns(name), self.head_nouns(other)
        if {wording.stemmer.stem(one) for one in heads} & {
            wording.stemmer.stem(one) for one in other_heads
        }:
            return True
        return any(
            self.meaning.likeness(one, two) >= self.meaning.settles_it
            for one in heads
            for two in other_heads
        )

    def far_apart(self, name: str, other: str) -> bool:
        """
        :param name: One name.
        :param other: The other.
        :return: Whether the two are further apart in meaning than the matcher reads the
            wording for. The head-noun rule only ever refuses more, so it leaves this to
            meaning.
        """
        return self.meaning.far_apart(name, other)
