(* ONNX Tensor API *)

module S = Tensor__Shape
module I = Tensor__Index
module T = Tensor__Tensor
module Sc = Scalar__Scalar

type 'a tensor = 'a T.tensor


let dim = T.dim
let shape t = t.T.shape


let scalar v =
  T.{
    shape = [] ;
    value = fun _ -> v ;
  }

exception Invalid_index = T.Invalid_index
let mem = T.mem
let get = T.get

let (.%[]) t k = get [k] t
let (.%[;..]) t ks = get (Array.to_list ks) t

let vector vs =
  let d = Array.of_list vs in
  let n = Array.length d in
  if n = 0 then invalid_arg "Tensor.vector" ;
  T.{
    shape = [ n ] ;
    value = (function [ k ] -> d.(k) | _ -> raise Invalid_index) ;
  }

let matrix vs =
  let d = Array.map Array.of_list @@ (Array.of_list vs) in
  let n = Array.length d in
  if n = 0 then invalid_arg "Tensor.matrix" ;
  let m = Array.length d.(0) in
  if m = 0 then invalid_arg "Tensor.matrix" ;
  if Array.exists (fun r -> Array.length r <> m) d then
    invalid_arg "Tensor.matrix" ;
  T.{
    shape = [ n ; m ] ;
    value = (function [ i ; j ] -> d.(i).(j) | _ -> raise Invalid_index) ;
  }

let tensorthreed vvs =
  let d = Array.map (fun inner_list_list ->  Array.map Array.of_list @@ (Array.of_list inner_list_list) (* 2D conversion *))@@(Array.of_list vvs) in 
  let n = Array.length d in
  if n = 0 then invalid_arg "Tensor.tensorthreed" ;
  let m = Array.length d.(0) in
  if m = 0 then invalid_arg "Tensor.tensorthreed" ;
  let l = Array.length d.(0).(0) in
  if l = 0 then invalid_arg "Tensor.tensorthreed" ;
  if Array.exists (fun r -> Array.length r <> m) d then
    invalid_arg "Tensor.tensorthreed" ;
  T.{
    shape = [ n ; m ; l ] ;
    value = (function [ i ; j ; k] -> d.(i).(j).(k) | _ -> raise Invalid_index) ;
  }

let tensorfourd vvvs = 
  let d =
    Array.map (fun lll -> (* lll is an 'a list list list *)
      Array.map (fun ll -> (* ll is an 'a list list *)
        Array.map Array.of_list (Array.of_list ll) (* Converts 'a list list to 'a array array *)
      ) (Array.of_list lll) (* Converts 'a list list list to 'a list list array *)
    ) (Array.of_list vvvs) (* Converts 'a list list list list to 'a list list list array *)
  in
 
  let d1 = Array.length d in
  if d1 = 0 then invalid_arg "Tensor.tensorfourd" ; 

  let d2 = Array.length d.(0) in
  if d2 = 0 then invalid_arg "Tensor.tensorfourd" ; 

  let d3 = Array.length d.(0).(0) in
  if d3 = 0 then invalid_arg "Tensor.tensorfourd" ; 
  let d4 = Array.length d.(0).(0).(0) in
  if d4 = 0 then invalid_arg "Tensor.tensorfourd" ; 

  if Array.exists (fun d1_element -> Array.length d1_element <> d2) d then
    invalid_arg "Tensor.tensorfourd" ; 

  T.{
    shape = [d1; d2; d3; d4];
    value = (function
      | [i; j; k; l] -> d.(i).(j).(k).(l)
      | _ -> raise Invalid_index 
    )
  }

  

let pretty pp fmt (t : 'a tensor) =
  match t.shape with
  | [] -> pp fmt @@ t.value []
  | [n] ->
      begin
        Format.fprintf fmt "[" ;
        for i = 0 to n-1 do
          Format.fprintf fmt " %a" pp @@ t.value [i]
        done ;
        Format.fprintf fmt " ]" ;
      end
  | [n;m] ->
      begin
        Format.fprintf fmt "@[<hv 0>" ;
        for i = 0 to n-1 do
          Format.fprintf fmt "[" ;
          for j = 0 to m-1 do
            Format.fprintf fmt " %a" pp @@ t.value [i;j]
          done ;
          Format.fprintf fmt " ]@," ;
        done ;
        Format.fprintf fmt "@]" ;
      end
  | [n;m;l] ->
    begin
      Format.fprintf fmt "@[<hv 0>" ;
      for i = 0 to n-1 do
        Format.fprintf fmt "(%d) [@," i ; (* Indicate slice index *)
        for j = 0 to m-1 do
           Format.fprintf fmt "[" ;
          for k = 0 to l-1 do
            Format.fprintf fmt " %a" pp @@ t.value [i;j;k]
          done;
            Format.fprintf fmt " ]@," ;
        done ;
        Format.fprintf fmt " ]@," ;
      done ;
      Format.fprintf fmt "@]" ;
    end
  
  | [d1; d2; d3; d4] ->
  begin
    Format.fprintf fmt "@[<v 0>" ; (* Overall vertical box for the entire 4D tensor *)

    for i = 0 to d1-1 do 
     
      if i > 0 then Format.fprintf fmt "@,@," ;
      Format.fprintf fmt "Hyper-slice %d (Dimension 1 = %d):@," i i;

      Format.fprintf fmt "@[<hv 2>";
      for j = 0 to d2-1 do 
        Format.fprintf fmt "(%d) [@," j ; 

        for k = 0 to d3-1 do 
          Format.fprintf fmt "  [" ;

          for l_idx = 0 to d4-1 do 
            Format.fprintf fmt " %a" pp @@ t.value [i;j;k;l_idx]
          done;

          Format.fprintf fmt " ]@," ;
        done ;

        Format.fprintf fmt "  ]@," ;
      done ;
    
      Format.fprintf fmt "@]" ; 
    done ;

    Format.fprintf fmt "@]" ; 
  end

  | _ -> invalid_arg "Tensor.pretty"


let pp_scalar fmt s =
  match s with
  | Sc.IsUInt8 v   -> Format.fprintf fmt "%d" v
  | Sc.IsInt16 v   -> Format.fprintf fmt "%d" v
  | Sc.IsUInt16 v  -> Format.fprintf fmt "%d" v

  (* These are mapped to OCaml's 'int32' *)
  | Sc.IsInt32 v   -> Format.fprintf fmt "%ld" v
  | Sc.IsUInt32 v  -> Format.fprintf fmt "%ld" v 

  (* These are mapped to OCaml's 'int64' *)
  | Sc.IsInt64 v   -> Format.fprintf fmt "%LdL" v
  | Sc.IsUInt64 v  -> Format.fprintf fmt "%LdL" v 

  (* These are mapped to OCaml's 'float' *)
  | Sc.IsFloat32 v -> Format.fprintf fmt "%f" v
  | Sc.IsFloat64 v -> Format.fprintf fmt "%.15f" v



let pretty_scalar fmt t = pretty pp_scalar fmt t

let add = Opadd__Add.add