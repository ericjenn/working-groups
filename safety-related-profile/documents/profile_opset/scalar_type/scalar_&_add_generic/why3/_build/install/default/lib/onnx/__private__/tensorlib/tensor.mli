(** ONNX Tensor OCaml Interface *)

type 'a tensor 

val dim : 'a tensor -> int
val shape : 'a tensor -> int list

val scalar : 'a -> 'a tensor
val vector : 'a list -> 'a tensor
val matrix : 'a list list -> 'a tensor
val tensorthreed : 'a list list list -> 'a tensor
val tensorfourd : 'a list list list list -> 'a tensor

exception Invalid_index

val mem : int list -> 'a tensor -> bool
val get : int list -> 'a tensor -> 'a
val (.%[]) : 'a tensor -> int -> 'a
val (.%[;..]) : 'a tensor -> int array -> 'a

val pretty : (Format.formatter -> 'a -> unit) -> Format.formatter -> 'a tensor -> unit
val pretty_int32 : Format.formatter -> (Scint32__ScalarInt32.scalar_i32 tensor) -> unit
val pretty_float32 : Format.formatter -> (Scfloat32__ScalarFloat32.scalar_f32 tensor) -> unit

val op_add_int32 :
  (Scint32__ScalarInt32.scalar_i32 tensor) ->
  (Scint32__ScalarInt32.scalar_i32 tensor) ->
  (Scint32__ScalarInt32.scalar_i32 tensor)

val op_add_float32 :
  (Scfloat32__ScalarFloat32.scalar_f32 tensor) ->
  (Scfloat32__ScalarFloat32.scalar_f32 tensor) ->
  (Scfloat32__ScalarFloat32.scalar_f32 tensor)


          
