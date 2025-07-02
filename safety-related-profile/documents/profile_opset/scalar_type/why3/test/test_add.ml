
open Int32
module Sc = Scalar__Scalar

let pp_real fmt v = Format.fprintf fmt "%3.2f" v 
let pp_bool fmt v = Format.fprintf fmt "%b" v
let pp_str fmt v = Format.fprintf fmt "%s" v
let pp_cplx fmt (c : Complex.t) =
  Format.fprintf fmt "(%g%s%gi)" c.Complex.re (if c.Complex.im >= 0.0 then "+" else "") c.Complex.im



let one_as_int32 = Int32.of_int 1
let two_as_int32 = Int32.of_int 2
let three_as_int32 = Int32.of_int 3

let int32_one = Sc.IsInt32 one_as_int32
let int32_two = Sc.IsInt32 two_as_int32
let int32_val = Sc.IsInt32 three_as_int32

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


let resulttwod = Tensor.add x y
let resulttwodtwo = Tensor.add x z
let resulttwodthree = Tensor.add y z


let float32_one   = Sc.IsFloat32 1.0
let float32_two   = Sc.IsFloat32 2.0
let float32_three = Sc.IsFloat32 3.0


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


let result_float1 = Tensor.add x_f y_f
let result_float2 = Tensor.add x_f z_f 


let print_named_tensor name t =
  Format.printf "%s:@\n" name;
  Tensor.pretty_scalar Format.std_formatter t;
  Format.print_newline ()
;; 


(* 2. Create ONE main execution block to run all tests *)
let () =
  Format.printf "\n--- Int32 Tensors ---\n";
  print_named_tensor "X" x;
  print_named_tensor "Y" y;
  print_named_tensor "Z" z;
  print_named_tensor "Result2D1" resulttwod;
  print_named_tensor "Result2D2" resulttwodtwo;
  print_named_tensor "Result2D3" resulttwodthree;

  Format.printf "\n--- Float32 Tensors ---\n";
  print_named_tensor "X (float)" x_f;
  print_named_tensor "Y (float)" y_f;
  print_named_tensor "Z (float)" z_f;
  print_named_tensor "Result (X+Y float)" result_float1;
  print_named_tensor "Result (X+Z float)" result_float2 
;;

