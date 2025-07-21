(** ONNX Tensor OCaml Interface *)
module Sc = Scalar__Scalar
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
val pretty_scalar : Format.formatter -> Sc.scalar tensor -> unit

val add : Sc.scalar tensor -> Sc.scalar tensor -> Sc.scalar tensor 


          
