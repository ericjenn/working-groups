type scalar =
  | IsUInt8 of (int)
  | IsInt16 of (int)
  | IsUInt16 of (int)
  | IsInt32 of (int32)
  | IsUInt32 of (int32)
  | IsInt64 of (int64)
  | IsUInt64 of (int64)
  | IsFloat32 of (float)
  | IsFloat64 of (float)

exception Mismatched_Types

let add (s1: scalar) (s2: scalar) : scalar =
  match (s1, s2) with
  | (IsUInt8 v1, IsUInt8 v2) -> IsUInt8 (v1 + v2)
  | (IsInt16 v1, IsInt16 v2) -> IsInt16 (v1 + v2)
  | (IsUInt16 v1, IsUInt16 v2) -> IsUInt16 (v1 + v2)
  | (IsInt32 v1, IsInt32 v2) -> IsInt32 (Int32.add v1 v2)
  | (IsUInt32 v1, IsUInt32 v2) -> IsUInt32 (Int32.add v1 v2)
  | (IsFloat32 v1, IsFloat32 v2) -> IsFloat32 (v1 +. v2)
  | (IsFloat64 v1, IsFloat64 v2) -> IsFloat64 (v1 +. v2)
  | (IsInt64 v1, IsInt64 v2) -> IsInt64 (Int64.add v1 v2)
  | (IsUInt64 v1, IsUInt64 v2) -> IsUInt64 (Int64.add v1 v2)
  | _ -> assert false (* absurd *)

