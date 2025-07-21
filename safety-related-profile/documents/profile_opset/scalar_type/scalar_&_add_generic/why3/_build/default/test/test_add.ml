(* Import your extracted modules *)
open Int32

(* Create Int32 values *)
let one_as_int32   = Int32.of_int 1
let two_as_int32   = Int32.of_int 2
let three_as_int32 = Int32.of_int 3

(* Wrap them into your scalar type *)
let int32_one = one_as_int32
let int32_two = two_as_int32
let int32_val = three_as_int32

(* Int32 tensors *)
let x = Tensor.matrix [
  [int32_one; int32_one];
  [int32_one; int32_one]
]

let y = Tensor.matrix [
  [int32_two; int32_two];
  [int32_two; int32_two]
]

let z = Tensor.matrix [
  [int32_val; int32_val];
  [int32_val; int32_val]
]

(* Perform tensor addition for Int32 *)
let resulttwod      = Tensor.op_add_int32 x y
let resulttwodtwo   = Tensor.op_add_int32 x z
let resulttwodthree = Tensor.op_add_int32 y z

(* Float32 values *)
let float32_one   = 1.0
let float32_two   = 2.0
let float32_three = 3.0

(* Float32 tensors *)
let x_f = Tensor.matrix [
  [float32_one; float32_one];
  [float32_one; float32_one]
]

let y_f = Tensor.matrix [
  [float32_two; float32_two];
  [float32_two; float32_two]
]

let z_f = Tensor.matrix [
  [float32_three; float32_three];
  [float32_three; float32_three]
]

(* Perform tensor addition for Float32 *)
let result_float1 = Tensor.op_add_float32 x_f y_f
let result_float2 = Tensor.op_add_float32 x_f z_f 

(* Float32 overflow test *)
let float_max   = 3.4e38
let float_large = 1e38

let x_f_over = Tensor.scalar float_max
let y_f_over = Tensor.scalar float_large

let result_float_over = Tensor.op_add_float32 x_f_over y_f_over
(* Expect: +inf or overflow behavior in float32 *)

(* Int32 overflow test *)
let int_max = Int32.of_string "2147483647"  (* Max Int32: 2^31 - 1 *)
let int_one = Int32.of_int 1

let x_i_over = Tensor.scalar int_max
let y_i_over = Tensor.scalar int_one

let result_int_over = Tensor.op_add_int32 x_i_over y_i_over

(* Printing utility *)
let print_named_tensor_int32 name t =
  Format.printf "%s:@\n" name;
  Tensor.pretty_int32 Format.std_formatter t;
  Format.print_newline ()
;;

(* Printing utility *)
let print_named_tensor_float32 name t =
  Format.printf "%s:@\n" name;
  Tensor.pretty_float32 Format.std_formatter t;
  Format.print_newline ()
;;

(* Main execution block *)
let () =
  Format.printf "\n--- Int32 Tensors ---\n";
  print_named_tensor_int32 "X" x;
  print_named_tensor_int32 "Y" y;
  print_named_tensor_int32 "Z" z;
  print_named_tensor_int32 "Result2D1" resulttwod;
  print_named_tensor_int32 "Result2D2" resulttwodtwo;
  print_named_tensor_int32 "Result2D3" resulttwodthree;

  Format.printf "\n--- Float32 Tensors ---\n";
  print_named_tensor_float32 "X (float)" x_f;
  print_named_tensor_float32 "Y (float)" y_f;
  print_named_tensor_float32 "Z (float)" z_f;
  print_named_tensor_float32 "Result (X+Y float)" result_float1;
  print_named_tensor_float32 "Result (X+Z float)" result_float2;

  Format.printf "\n--- Float32 Overflow Tensor ---\n";
  print_named_tensor_float32 "Overflow float" result_float_over;

  Format.printf "\n--- Int32 Overflow Tensor ---\n";
  print_named_tensor_int32 "Overflow int" result_int_over
;;
