# The Warsaw pipeline

A labelled scan of a room goes in; an annotated, hierarchical world comes out. The run
asks a vision-language model which class each of the scan's labels means, who owns the
faces two labels both claim, and what each body is, then cuts the scan into one body per
object and mounts the parts into their wholes.

    python -m experiments.warsaw.pipeline.pipeline

That is the whole invocation. Nothing is passed on a command line: everything a run can
be told is a default in `pipeline/settings.py`, so a run can be read back from the file
rather than from someone's shell history.

On some machines the renders come back blank and the run stops with a
`BlankRenderError`; see below for what to do about that.

## Where the scan goes

    experiments/src/experiments/warsaw/dataset/<scene name>/<anything>.ply

One directory per scene, holding exactly one `.ply`. `WarsawScene.from_directory` globs
`*.ply` and refuses a directory with none or with more than one, so a second mesh beside
the scene is an error rather than a coin toss.

`dataset/` is not in the repository — it is gitignored, along with `pipeline_runs/` — so
a fresh clone has no such directory and you make it yourself. Which scene is read is
`PipelineSettings.scene_directory`, whose default is
`dataset/kitchenlab_new_mesh_agreement_dataset`.

## What the scan has to be

A single mesh whose faces carry the labels, written as **one integer face property per
class, named for the class**:

    element face 1749164
    property list uchar int vertex_indices
    property int floor
    property int cabinet
    property int drawer
    ...

The value of a face's `cabinet` property is *which* cabinet the face belongs to; `0`
means no cabinet covers it. So one property per class, one instance number per face, and
a face may belong to objects of several classes at once — a drawer front is part of the
drawer and of the cabinet holding it. Sorting that out is what the pipeline does.

Two things about the file are easy to lose:

- **Only the PLY reader keeps these properties**, and only when the mesh is loaded
  unprocessed. Re-exporting the scan through most tools drops the label properties, or
  welds vertices and renumbers the faces the labels were written for. Either way the
  labels no longer describe the mesh. The pipeline checks both and says so.
- **The scan is read y-up** and rolled onto the world's z
  (`WarsawScene.world_T_source`). A scan written z-up arrives lying on its side.

## What else a run needs

| Variable | What for |
|---|---|
| `OPENROUTER_API_KEY` | Every question goes to OpenRouter. A run is about a hundred calls; on the default model that is a few cents. |
| `SEMANTIC_DIGITAL_TWIN_DATABASE_URI` | A `postgresql+psycopg://` URI. Each run gets a schema of its own under it, so no run inherits another's tables. |

Without a database, set `PipelineSettings.persist = False`. The run then stops after the
split's report, since every step past it reads a world back.

Somewhere to draw is the third requirement. Usually the display you are already sitting
at will do; the next section is for when it will not.

## When the renders come back blank

Every question comes with pictures. trimesh draws them by opening a window, keeping it
hidden, and reading the picture back out of that window's colour buffer.

Whether a hidden window has a colour buffer worth reading is up to the graphics stack.
Nothing obliges it to allocate or to draw into a window that was never shown, and where
it does not, the read comes back blank instead of failing — trimesh warns of exactly this
in `render_scene`. A desktop session with a working driver generally does keep the buffer,
which is why the plain invocation above is usually all you need; a machine with no display
at all, a virtual one, or a software renderer often does not.

When it does not, the run draws nothing but black, measures every object as invisible,
and still pays for a hundred questions about black pictures. Both ways out below give
the window somewhere real to be drawn.

The first blank render raises `BlankRenderError` and stops the run before anything is
asked. Two ways out of it:

- **`xvfb-run -a python -m experiments.warsaw.pipeline.pipeline`** gives the run a
  display of its own. This is the fallback that works anywhere, including over ssh, and
  needs the `xvfb` package installed.
- **`PipelineSettings.headless = False`** draws into a window you can see, on a machine
  that has a display to put one on. Slower, and it fills your screen with renders, but it
  is the quickest way to tell whether the renderer works at all.

## What a run leaves behind

A directory under `pipeline_runs/`, named for when the run started, holding every
question as it was put, every answer as it came back, the renders that went with them,
the bodies' meshes, and at the end:

- `report.md` — what was measured, what was asked, what was built, and what it cost.
- `inspect_world.py` — opens the world the run wrote.

Plus a database schema named for the same run. `RunSchema.drop` throws it away, beside
deleting the directory.

## The steps, in order

1. **Prepare** — put the ontology back as it is committed, rebuild the ORM, give the run
   a schema.
2. **Measure** — how the labelled objects meet, and one render per label.
3. **Vocabulary** — which class of the ontology each label means.
4. **Measure again**, now knowing the classes: what the ontology settles, what is left
   open.
5. **Adjudicate** — whose the contested faces are, and which whole each part belongs to.
6. **Split** — one body per object, written to the database.
7. **Classify** and **mount** — what each body is, and the hierarchy.

Steps 3, 5 and 7 are the ones that cost money. Every reply is kept beside the question
it answered, and `PipelineSettings.reuse_answers` reads a kept reply rather than asking
again, which re-reads a run without spending anything on it.
