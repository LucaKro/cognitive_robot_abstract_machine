#!/usr/bin/env python3

import os
import logging
from pathlib import Path
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.spatial_computations.raytracer import RayTracer


def main():
    # Disable critical logging to keep output clean
    logging.disable(logging.CRITICAL)

    print("Locating resources...")
    # Find the root directory of the package
    try:
        root = Path(__file__).parent.parent
    except Exception as e:
        # Fallback if running from root of repo
        # get_semantic_digital_twin_directory_root often expects to find specific markers
        # If it fails, we might be in the repo root.
        # Let's try to assume 'semantic_digital_twin' is in 'src/semantic_digital_twin' or similar if checking fails.
        # But based on usage in docs, it should work if we are in a valid environment.
        print(f"Warning: Could not determine root using utils: {e}")
        root = os.getcwd()

    # Path to the example kitchen URDF

    kitchen_urdf = os.path.join(
        root, "semantic_digital_twin", "resources", "urdf", "kitchen.urdf"
    )

    # If not found there, try relative path if we are in repo root
    if not os.path.exists(kitchen_urdf):
        potential_path = os.path.join(
            "semantic_digital_twin", "resources", "urdf", "kitchen.urdf"
        )
        if os.path.exists(potential_path):
            kitchen_urdf = potential_path

    if not os.path.exists(kitchen_urdf):
        print(f"Error: Could not find kitchen.urdf at {kitchen_urdf}")
        print(
            "Please ensure you are running this from the repository root or correct environment."
        )
        return

    print(f"Loading world from {kitchen_urdf}...")
    # Parse the URDF into a World object
    world = URDFParser.from_file(kitchen_urdf).parse()

    print(f"World loaded: {world.name}")
    print(f"Bodies: {len(world.bodies)}")

    print("Initializing RayTracer and rendering scene...")
    # Initialize RayTracer with the loaded world
    rt = RayTracer(world)

    # Update the scene with the world state
    rt.update_scene()

    print("Opening viewer window...")
    # Show the scene
    rt.scene.show()


if __name__ == "__main__":
    main()
