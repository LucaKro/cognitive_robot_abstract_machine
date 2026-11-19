#134 (screw_axis_type, on main): adds krrood_spatial_math ScrewAxis (rotation_per_unit_position, translation_per_unit_position, unit_direction, to_twist), 11 tests. Independent of the #120 stack by request.
Decided (Luca): longer field names; create_*_screw_axis constructors stay on ScrewAxisConnection.
Next: #120 adopts ScrewAxis for its screw_axis field (plus ORM mapping), then sync #130/#132/#133 and the version 2 branch. Fork lacks the new-feature label.
