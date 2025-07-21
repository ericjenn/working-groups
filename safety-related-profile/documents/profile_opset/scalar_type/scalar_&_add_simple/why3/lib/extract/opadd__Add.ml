let add (a: Scalar__Scalar.scalar Tensor__Tensor.tensor)
        (b: Scalar__Scalar.scalar Tensor__Tensor.tensor) :
  Scalar__Scalar.scalar Tensor__Tensor.tensor =
  let result_shape = a.Tensor__Tensor.shape in
  { Tensor__Tensor.shape = result_shape; Tensor__Tensor.value =
    (fun (i: (int) list) ->
       (let val_a = a.Tensor__Tensor.value i in
        let val_b = b.Tensor__Tensor.value i in
        Scalar__Scalar.add val_a val_b)) }

