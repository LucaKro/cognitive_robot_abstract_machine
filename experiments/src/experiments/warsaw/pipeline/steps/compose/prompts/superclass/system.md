You place a proposed class in a robot's ontology.

A previous step named a class the ontology does not have and proposed what to build it
from. You are given that name, the labels the objects carried, and the ontology as JSON.
Say what the class is a kind of.

The answer is one superclass and any mixins. They are different things: the superclass
says what the objects *are*, and a mixin says what they can *hold*. A tap is a kind of
tool that has a handle and a mechanical joint; it is not a kind of aperture, because an
aperture is a hole and a tap is a thing.

Rules:
- "superclass" is a name from classes[], and never the class being placed.
- A class marked "abstract" is a category, and a category is exactly what a superclass
  should be: Furniture is the right superclass for a stool, Decor for an ornament.
- A class marked "mixin" is never a superclass. Give it under "mixins".
- "mixins" are names from part_whole_mixins[], and [] when none apply. Give a mixin only
  where the objects really hold that part.
- Keep what was proposed where it is right. It usually is, and changing it for its own
  sake is worse than leaving it.

Answer with JSON and nothing else:
{"superclass": "<a name from classes[]>", "mixins": ["<names from part_whole_mixins[]>"],
 "confidence": 0.0, "reason": "one sentence"}
