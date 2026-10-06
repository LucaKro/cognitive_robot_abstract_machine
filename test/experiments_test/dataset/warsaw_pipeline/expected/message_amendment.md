## The proposal
Give CounterTop the mixin HasDrawers, which introduces: drawers -> Drawer.

## The class as it stands
CounterTop(Furniture, HasSupportingSurface, HasSink)
  part sink -> Sink, mounted with add()
  contains objects -> HasRootBody (many), mounted with add_object()
  supports supporting_surface -> Region, mounted with add_supporting_surface()

## The part
Drawer(Furniture, HasCaseAsRootBody, HasHandle, HasMechanicalJoint)
  part mechanical_joint -> MechanicalJoint, mounted with add()
  part handle -> Handle, mounted with add()
  contains objects -> HasRootBody (many), mounted with add_object()
  supports supporting_surface -> Region, mounted with add_supporting_surface()

## What those parts hold in turn
Sink(HasRootBody)

## What was measured
In one scanned room, objects labelled countertop were read as CounterTop, and objects labelled drawer as Drawer. They share faces over 7 measured pairs, 1234 shared faces in all.