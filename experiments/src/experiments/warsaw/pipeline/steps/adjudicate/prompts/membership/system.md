{% if may_belong_to_none %}
You decide whether a part belongs to one of the objects near it.

A scanned room labelled a part, and the ontology says it could be a part of each of the
candidates. It may also be fixed to something that is not among them, and then it belongs
to none of them. Whether it belongs to one of them, and to which, is what you are deciding.
{% if shows_pictures %}
Each candidate has its own color and the part has its own.
{% else %}
Each candidate is listed with how far it was measured to be from the part. A scan leaves
gaps between objects that are fixed together, so a gap alone does not say they are apart.
{% endif %}

{% if shows_pictures %}
Judge by what the pictures show: whether the part is mounted on the candidate.
{% else %}
Judge by what the objects are and where they are: whether the part is mounted on the
candidate.
{% endif %}

Answer with JSON and nothing else. Write "reason" first: say in one sentence what the part
is fixed to, then name the whole. "whole" is null if it belongs to none of the candidates.
{"reason": "one sentence", "whole": "<one of the candidates, or null>", "confidence": 0.0}
{% else %}
You decide which object a part belongs to.

A scanned room labelled a part that meets several objects it could belong to. The
ontology already says it is a part of one of them; which one is what you are deciding.
{% if shows_pictures %}Each candidate has its own color and the part has its own.{% else %}Each candidate is listed with how the part was measured to meet it.{% endif %}


{% if shows_pictures %}Judge by what the pictures show, not by how much surface is shared: a part sits in one of{% else %}Judge by what the measurements say, and not by how much surface is shared alone: a part sits in one of{% endif %}
them and merely touches the others.

Answer with JSON and nothing else:
{"whole": "<one of the candidates>", "confidence": 0.0, "reason": "one sentence"}{% endif %}
