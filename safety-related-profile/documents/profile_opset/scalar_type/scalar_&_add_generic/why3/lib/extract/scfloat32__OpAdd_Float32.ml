module Add =
struct
  let tensor_add (a: (float) Tensor__Tensor.tensor)
                 (b: (float) Tensor__Tensor.tensor) :
    (float) Tensor__Tensor.tensor =
    let result_shape = a.Tensor__Tensor.shape in
    { Tensor__Tensor.shape = result_shape; Tensor__Tensor.value =
      (fun (i: (int) list) ->
         Scfloat32__ScalarFloat32.add_float32 (a.Tensor__Tensor.value i)
         (b.Tensor__Tensor.value i)) }
end

