module Add =
struct
  let tensor_add (a: (int32) Tensor__Tensor.tensor)
                 (b: (int32) Tensor__Tensor.tensor) :
    (int32) Tensor__Tensor.tensor =
    let result_shape = a.Tensor__Tensor.shape in
    { Tensor__Tensor.shape = result_shape; Tensor__Tensor.value =
      (fun (i: (int) list) ->
         Scint32__ScalarInt32.add_int32 (a.Tensor__Tensor.value i)
         (b.Tensor__Tensor.value i)) }
end

