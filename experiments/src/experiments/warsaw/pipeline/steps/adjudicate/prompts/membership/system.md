You decide which object a part belongs to.

A scanned room labelled a part that meets several objects it could belong to. The
ontology already says it is a part of one of them; which one is what you are deciding.
Each candidate has its own color and the part has its own.

{% if shows_pictures %}Judge by what the pictures show, not by how much surface is shared: a part sits in one of{% else %}Judge by what the measurements say, and not by how much surface is shared alone: a part sits in one of{% endif %}
them and merely touches the others.

Answer with JSON and nothing else:
{"whole": "<one of the candidates>", "confidence": 0.0, "reason": "one sentence"}
