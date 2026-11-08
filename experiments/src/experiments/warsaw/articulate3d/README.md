# Evaluating the Warsaw pipeline on Articulate3D

Articulate3D annotates the parts of articulated objects (cabinets, doors, drawers,
handles, ...) on the meshes of ScanNet++ scenes, together with the part hierarchy. This
package turns that annotation into the pipeline's input and into ground truth, and scores
finished runs against it.

The ground truth is built from Articulate3D's annotation and from a **protocol** a person
writes once, before any run on the evaluated scenes is looked at. Nothing in it comes from
inference rules or from what a run produced.

## What is fixed by hand, and when

| Fixed before the evaluated runs | Never allowed |
| --- | --- |
| The ontology (a pinned commit; extensions decided on train scenes only) | Changing anything below after looking at results on the evaluated scenes |
| The protocol: what every part name means, parts classified on sight, field choices | Decisions about a single scene that are not written into the protocol |
| Prompts and pipeline settings, tuned on train scenes | Anything derived from a run's output |

Freeze all of it with one git tag before running the evaluated scenes. The validation
scenes (`splits/val.txt`) are locked until then: converting, running or scoring one
raises `ValidationSceneLockedError` unless the checkout has no local changes and its
commit carries a tag starting with `articulate3d-frozen-`. Train scenes
(`splits/train.txt`) are always open.

## Workflow

1. **List the part names** the annotators used and add them to the protocol, which is
   kept with this package as `protocol.json`. Every new name is added as `undecided`:

   ```bash
   python -m experiments.warsaw.articulate3d.labels --dataset <articulate3d> \
       --scannetpp <scannet++ data> \
       --protocol experiments/src/experiments/warsaw/articulate3d/protocol.json
   ```

2. **Rule on every name** in `protocol.json`. The commands below read that file unless
   `--protocol` names another:
   - `class`: the name means one class of the ontology (`semantic_class`);
   - `ambiguous`: the name covers different kinds of things, so these parts are judged as
     objects and as ends of relations, but not on their class;
   - `ignored`: the evaluation leaves these parts out.

   Optionally, classify parts on sight in `part_classes`; this overrides what the part's
   name means. Two people should classify without seeing the part names or any run.
   Where a whole could hold a part through several fields, add a `field_choices` entry.
   Running the command again adds new names and keeps every ruling already made.

3. **Build the ground truth** of every scene. This fails while a name is undecided, while
   the protocol names a class the ontology lacks, or while a field choice is missing:

   ```bash
   python -m experiments.warsaw.articulate3d.ground_truth --dataset <articulate3d> \
       --scannetpp <scannet++ data> --output ground_truth/
   ```

4. **Convert each scene** into the pipeline's input. The part hierarchy is stripped from
   the labels, and instance numbers are random:

   ```bash
   python -m experiments.warsaw.articulate3d.convert --dataset <articulate3d> \
       --scannetpp <scannet++ data> --scene <scene id> --output <scene directory> \
       [--disclosure words|neutral] [--overlap disjoint|subtree] \
       [--spelling as-written|corrected] [--seed N]
   ```

   - `--disclosure neutral` writes labels such as `label_03`, so the model learns nothing
     from the annotators' words. This tests what the pictures alone tell it.
   - `--overlap subtree` makes each part's label cover the parts below it as well. This
     recreates the overlapping labels a real scan produces, so the ownership step runs.
   - `--spelling corrected` fixes the annotators' misspelt part names before they become
     labels; the default keeps them as written. Spacing and instance numbers are tidied
     either way.

   ScanNet++ meshes must be the v1 meshes the annotations were made on.

5. **Run the pipeline** on each converted scene. Point
   `PipelineSettings.scene_directory` at it, and run each scene several times. Which
   model is asked, whether it sees pictures and how it proposes classes are settings;
   [the pipeline's README](../README.md#running-a-version-of-the-pipeline) says how to
   run each version.

6. **Evaluate the runs**: score each, judge its answers, judge the baselines and what the
   rule allows on the same scenes, and write the tables for a document. Point it at the
   runs, at a batch, or at `pipeline_runs/`:

   ```bash
   python -m experiments.warsaw.articulate3d.evaluate_runs <run, batch or directory> \
       --ground-truth ground_truth/ --measured measured/ --output tables/
   ```

   [The evaluation's README](../evaluation/README.md#tables-of-results) says what the
   tables hold. To score runs without the rest:

   ```bash
   python -m experiments.warsaw.articulate3d.score <run> [<run> ...] \
       --ground-truth ground_truth/ --summary summary.json
   ```

## What the score means

Bodies are paired with annotated parts one to one, by the faces they share (at least half
of the faces either covers, by default). Class and position play no part in the pairing,
so a misclassified body is still found and counted as a misclassification.

| Number | Meaning |
| --- | --- |
| parts found | parts paired with a body |
| fragmented parts / merged bodies | parts split over several bodies / bodies covering several parts |
| bodies outside | bodies lying on faces the annotation does not cover; not judged |
| class accuracy | found parts given exactly their class |
| classes recognised | the same, out of every part whose class is judged, found or not |
| class-hierarchy precision / recall | partial credit for an answer on the right branch of the class hierarchy |
| answers standing for no class | generated classes with no single class of the ontology as their base |
| part-whole precision | parts the run hung from another part that hang from their annotated whole |
| part-whole recall | relations of the part hierarchy the run made |
| representable relations | relations a field of either class can make: the most any run can make |
| representable recall | of those, the ones the run made |
| whole-chain precision / recall | partial credit for a part hung on the right branch, such as a handle hung from its door rather than from the plate on it |
| part-whole with classes precision / recall | relations made with both ends given exactly their class, out of the relations whose ends' classes are judged |

Part-whole relations are read from the tree the run built. A part counts as hung from the
nearest annotated part above it, past any body between that matches no part. A part the
annotation hangs from nothing counts against precision when the run hangs it from a part.
A class the run generated is judged as the one class of the ontology it derives
from, with mixins left out. A share with nothing to divide by is reported as `n/a`,
never as a perfect score.

## Baselines and what the rule allows

Both need every scene measured once as the pipeline measures it. Measuring converts the
scene with the default conversion (annotators' words, disjoint labels, spelling as
written), measures it, and keeps only a small file per scene:

```bash
python -m experiments.warsaw.articulate3d.measured_scene --dataset <articulate3d> \
    --scannetpp <scannet++ data> --output measured/ <scene id> [<scene id> ...]
```

A scene already measured is skipped. Measuring many scenes in one process lets memory
grow, so measure each scene in a process of its own, for instance through `xargs -n1`.

**What the rule allows.** With the protocol's classes in place of a model's readings,
the pipeline's candidate rule decides each relation of the part hierarchy: it takes the
true whole without asking, asks with the true whole among the candidates, or loses the
relation. The share taken or asked about, of the relations the ontology can hold, is the
most any model answering the rule's questions can reach:

```bash
python -m experiments.warsaw.articulate3d.rule_ceiling --measured measured/ \
    --ground-truth ground_truth/ [--largest-gap-to-a-whole 0.10] \
    [--largest-gap-counted-as-touching 0.01] [--no-confirm-wholes-within-the-gap] \
    [--output ceiling.json]
```

**Baselines.** Each baseline reads every label as a class and chooses every part's whole
without a model, builds the graph a run writes, and is scored like a run:

| Baseline | Label to class | Part to whole |
| --- | --- | --- |
| `embedding-and-geometry` | nearest class name by Sentence-BERT | the larger object the part touches most |
| `spelling-and-rule` | the class name spelled most alike | the pipeline's rule, each question answered by its first candidate |
| `embedding-and-rule` | nearest class name by Sentence-BERT | the pipeline's rule, as above: the pipeline without a model |
| `true-classes-and-rule` | the protocol's classes | the pipeline's rule, as above: the most the rule gives without a model |

```bash
python -m experiments.warsaw.articulate3d.baselines --measured measured/ \
    --ground-truth ground_truth/ --output baselines/ \
    [--baseline <name> ...] [--scenes <scene id> ...]
```

Each baseline's summary is written as `<name>.json` and holds the same counts as a
run's. Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`) is fetched from the
Hugging Face hub on first use. Restrict `--scenes` to the scenes of a set of runs to set
the baselines beside those runs.

**What a run's answers add.** A finished run is judged twice: as it is, and with every
question about wholes answered by its first candidate, the one touching the part along
the most edges. Everything else the run did stays, its readings of the labels included,
so the difference is the model's answers alone. The questions that may be answered with
none are counted by how they came out: true whole kept, true whole declined, another
taken, wrong whole avoided, wrong whole taken. The share that offered the true whole is
how often always taking a candidate would have been right, which the model's answers
have to beat for asking to pay off:

```bash
python -m experiments.warsaw.articulate3d.run_answers <run> [<run> ...] \
    --ground-truth ground_truth/
```

Each run's result is kept in its directory as `articulate3d_answers.json`, beside its
`articulate3d_score.json`.

## Not covered yet

- **Joints.** The pipeline mounts every part with a fixed connection, and the
  annotated joints are not evaluated.
- **Held-out ontology classes.** Testing class generation by hiding classes from the
  taxonomy the pipeline sees needs a change to the taxonomy export.
- **Confidence intervals.** Spread over scenes and over repeated runs is not computed yet.
