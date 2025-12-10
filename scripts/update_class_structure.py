import argparse
import json
import logging
from pathlib import Path

from semantic_digital_twin.semantic_annotations import semantic_annotations as sa_module

import sys

from semantic_digital_twin.semantic_annotations.mixins import HasBody
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationClassBuilder,
    SemanticAnnotationFilePaths,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

sys.path.insert(
    0, str(Path(__file__).parent.parent / "semantic_digital_twin" / "scripts")
)

from load_warsaw_scene import load_world


def build_class_lookup():
    """Build a name -> class lookup from the semantic_annotations module."""
    lookup = {}
    for name in dir(sa_module):
        obj = getattr(sa_module, name)
        if isinstance(obj, type):
            lookup[name] = obj
    return lookup


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info(f"Loading world from {args.world_dir}...")
    world: World = load_world(args.world_dir)

    with open(args.model_output_json, "r") as f:
        all_data = json.load(f)

    for body_group in range(len(all_data)):
        data = all_data[body_group]
        body_ids = data["body_ids"]
        colors = data["colors"]
        model_output = data["vlm_response"]["choices"][0]["message"]["content"]
        model_output = model_output.replace("```json", "").replace("```", "")
        logging.info("Model output content:\n%s", model_output)
        model_output = json.loads(model_output)
        objects = model_output["objects"]

        class_lookup = build_class_lookup()

        for obj in objects:
            if obj["confidence"] < 0.9:
                logging.warning(f"Skipping {obj} due to low confidence")
                continue

            cls_name = obj["classification"]["class"]

            if obj["classification"]["is_new_class"]:
                # Create a new class for the object
                superclass_name = obj["classification"]["superclass"]
                logging.info(
                    "Creating new class: %s with superclass %s",
                    cls_name,
                    superclass_name,
                )

                if superclass_name not in class_lookup:
                    logging.warning(
                        "Superclass '%s' not found in semantic_annotations",
                        superclass_name,
                    )
                    continue

                superclass = class_lookup[superclass_name]

                # Create new class with builder
                builder = SemanticAnnotationClassBuilder(
                    cls_name, template_name="dataclass_template.py.jinja"
                )

                cls = builder.add_base(superclass).build()
                builder.append_to_file(
                    SemanticAnnotationFilePaths.MAIN_SEMANTIC_ANNOTATION_FILE.value,
                    include_imports=True,
                )

                # Update lookup structure to avoid same class being created twice
                class_lookup[cls_name] = cls
                logging.info(
                    "Created class '%s' (subclass of '%s')", cls_name, superclass_name
                )
            else:
                cls = class_lookup[cls_name]

            # Create an instance for the object
            try:
                # kwargs for constructor
                kwargs = {}

                if issubclass(cls, HasBody):
                    # Subclasses of HasBody expect to be passed the body in the constructor -> Find the corresponding body in the world
                    obj_color = obj["highlight_color"]
                    obj_idx = colors.index(obj_color)
                    body_id = body_ids[obj_idx]
                    body = next(b for b in world.bodies if str(b.id) == body_id)
                    kwargs["body"] = body

                instance = cls(**kwargs)
                logging.info("Instance of class '%s' created: %s", cls_name, instance)
            except Exception as e:
                logging.warning("Could not instantiate class '%s': %s", cls_name, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_output_json", type=Path)
    parser.add_argument("world_dir", type=Path)
    args = parser.parse_args()
    main(args)
