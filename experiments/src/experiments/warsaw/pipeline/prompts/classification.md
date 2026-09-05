You are naming objects in a scanned room, using an ontology of classes.

Each picture shows the same room from a different side, with a few objects painted in
distinct colors and everything else left as it was scanned. Say what each painted object
is. You are told the label a previous step gave it and the class that label was mapped
to; both can be wrong about this particular object, which is what you are being asked.

Rules:
- Name every object you are given, once, by the name it is listed under.
- "class" is a name from the ontology's classes, or a name you propose.
- If you propose one, "is_new_class" is true and "superclass" is a class of the ontology.
- Judge the object, not the paint: the colors mark what to look at, nothing more.
- A class marked "abstract" cannot be given to an object. Name one of its subclasses,
  or propose a new class with it as the superclass.

Answer with JSON and nothing else:
{"objects": [{"name": "drawer_19", "class": "Drawer", "is_new_class": false,
              "superclass": "Furniture", "confidence": 0.0, "reason": "one sentence"}]}
