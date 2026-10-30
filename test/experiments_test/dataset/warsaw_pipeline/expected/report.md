# run

- scene: `dataset/kitchenlab_new_mesh_agreement_dataset/mesh_all_classes.ply`
- worlds: **198** annotated, 1 as it was split
- models: qwen/qwen3-vl-30b-a3b-instruct (vocabulary), qwen/qwen3-vl-30b-a3b-instruct (adjudication), qwen/qwen3-vl-30b-a3b-instruct (classification)

## What was measured

- 15 labelled objects over 6 measurable pairs, 4 of them sharing faces
- 1 sets of contested faces the ontology settled, 3 memberships with only one candidate

## What was asked

- 6 labels, 6 mapped to a class, 1 of them new
- 1 class patterns and 5 memberships adjudicated, 0 with problems
- 13 bodies named, 6 distinct classes

## What was built

- 13 bodies, 67979 faces between them, 0 faces still claimed twice
- 7 pairings carried past the split, 7 of them mounted

### 2 objects lost every face

- `cabinet_10` -> drawer_7 (827), handle_13 (69), drawer_6 (7), kitchen_island_1 (5)
- `cabinet_8` -> drawer_5 (1509), handle_14 (72), drawer_4 (8)

### Classes given

- 4 x `Drawer`
- 3 x `Cabinet`
- 2 x `Door`
- 2 x `Handle`
- 1 x `Floor`
- 1 x `KitchenIsland`

## Looking at it

```
python inspect_world.py
```
